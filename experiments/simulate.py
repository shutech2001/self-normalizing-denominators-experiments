from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import norm

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"

Z975 = float(norm.ppf(0.975))
PROBS = np.array([0.025, 0.5, 0.975])
HGRID = np.array(
    [
        0,
        0.25,
        0.5,
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
        512,
        1024,
        2048,
        4096,
        8192.0,
    ]
)


def generator(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *key]))


def rates(mask: np.ndarray, valid: np.ndarray | None = None) -> dict:
    if valid is None:
        valid = np.ones(mask.shape, dtype=bool)
    success = np.asarray(mask, bool) & valid
    B = len(success)
    p = float(success.mean())
    nv = int(valid.sum())
    return {
        "coverage": p,
        "mcse": float(np.sqrt(p * (1 - p) / B)),
        "failures": int(B - nv),
        "conditional_coverage": float(success.sum() / nv) if nv else np.nan,
    }


def summarize(x: np.ndarray, prefix: str = "") -> dict:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {
            prefix + k: np.nan
            for k in ("q025", "q50", "q975", "q025_mcse", "q50_mcse", "q975_mcse", "sd")
        }
    q = np.quantile(a, PROBS)
    # Independent replication batches, not a bootstrap of simulation output.
    groups = [g for g in np.array_split(a, 25) if len(g) >= 20]
    se = (
        np.std([np.quantile(g, PROBS) for g in groups], axis=0, ddof=1) / np.sqrt(len(groups))
        if len(groups) > 1
        else np.full(3, np.nan)
    )
    out = {prefix + k: float(v) for k, v in zip(("q025", "q50", "q975"), q)}
    out.update({prefix + k: float(v) for k, v in zip(("q025_mcse", "q50_mcse", "q975_mcse"), se)})
    out[prefix + "sd"] = float(np.std(a, ddof=1))
    return out


def paired(mask1, mask2):
    d = np.asarray(mask1, float) - np.asarray(mask2, float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def chol_logdet(s: np.ndarray) -> np.ndarray:
    """Batched Cholesky; preserve individual failures without ridge or deletion."""
    try:
        L = np.linalg.cholesky(s)
        return 2 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)
    except np.linalg.LinAlgError:
        out = np.full(len(s), np.nan)
        for i, ss in enumerate(s):
            try:
                out[i] = 2 * np.log(np.diag(np.linalg.cholesky(ss))).sum()
            except np.linalg.LinAlgError:
                pass
        return out


@dataclass
class LogPivot:
    n: int
    df: np.ndarray
    power: np.ndarray
    x: np.ndarray
    cdf: np.ndarray
    quantiles: np.ndarray
    mean: float
    variance: float
    grid_difference: float
    truncation_difference: float
    raw_negative_mass: float

    @staticmethod
    def make(n: int, df, power) -> "LogPivot":
        df = np.array(df, float)
        power = np.array(power, float)
        if len(df) != len(power) or np.any(df <= 0) or np.any(power <= 0):
            raise ValueError("Positive degrees of freedom and powers required.")
        mu = float(np.dot(power, digamma(df / 2) + np.log(2 / n)))
        va = float(np.dot(power**2, polygamma(1, df / 2)))
        span = max(8.0, 40 * np.sqrt(va))

        def grid(N, span):
            dx = span / N
            t = 2 * np.pi * np.fft.fftfreq(N, d=dx)
            logphi = np.zeros(N, dtype=complex)
            for nu, m in zip(df, power):
                k = nu / 2
                logphi += loggamma(k + 1j * m * t) - gammaln(k) - 1j * m * t * digamma(k)
            # Centered log pivot; inverse Fourier sign uses fft, not ifft.
            density = np.fft.fftshift(np.fft.fft(np.exp(logphi)).real) / span
            negative = float(np.maximum(-density, 0).sum() * dx)
            density = np.maximum(density, 0)
            xx = mu + (np.arange(N) - N / 2) * dx
            cc = cumulative_simpson(density, x=xx, initial=0)
            cc = np.maximum.accumulate(np.clip(cc, 0, None))
            cc /= cc[-1]
            good = np.r_[True, np.diff(cc) > 1e-15]
            inverse = PchipInterpolator(cc[good], xx[good], extrapolate=False)
            return xx, cc, np.asarray(inverse(PROBS)), negative

        x0, c0, q0, neg0 = grid(2**16, span)
        x, c, q, neg = grid(2**17, span)
        _, _, qw, negw = grid(2**18, 2 * span)
        dif = float(np.max(np.abs(q - q0)))
        trunc = float(np.max(np.abs(q - qw)))
        if dif > 2e-6 or trunc > 2e-6 or neg > 1e-8:
            raise ArithmeticError(f"Pivot grid check failed: {dif}, {trunc}, {neg}")
        return LogPivot(n, df, power, x, c, q, mu, va, dif, trunc, neg)

    def prob(self, log_ratio):
        return np.interp(log_ratio, self.x, self.cdf, left=0, right=1)

    def central(self, log_ratio):
        return (log_ratio >= self.quantiles[0]) & (log_ratio <= self.quantiles[2])


class Study:
    def __init__(
        self,
        out: Path,
        seed: int = 42,
        reps: int = 50_000,
        reference: int = 2_000_000,
        limit_reps: int = 1_000_000,
    ):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.reps = reps
        self.reference = reference
        self.limit_reps = limit_reps
        self.pivots = {}
        self.reference_audit = []
        self.timings = {}

    def pivot(self, n, df, power):
        key = (int(n), tuple(df), tuple(power))
        if key in self.pivots:
            return self.pivots[key]
        p = LogPivot.make(n, df, power)
        # The stream depends on the mathematical reference, not cache order.
        tag = [100, n, len(df)] + list(map(int, df)) + list(map(int, power))
        rng = generator(self.seed, *tag)
        logs = np.zeros(self.reference)
        for nu, m in zip(df, power):
            logs += m * np.log(rng.chisquare(nu, self.reference) / n)
        est = np.array([(logs <= qq).mean() for qq in p.quantiles])
        z = (est - PROBS) / np.sqrt(PROBS * (1 - PROBS) / self.reference)
        row = dict(
            n=n,
            df=list(map(int, df)),
            powers=list(map(int, power)),
            reference_reps=self.reference,
            q025=float(p.quantiles[0]),
            q50=float(p.quantiles[1]),
            q975=float(p.quantiles[2]),
            mean=p.mean,
            variance=p.variance,
            grid_difference=p.grid_difference,
            doubled_range_difference=p.truncation_difference,
            negative_mass=p.raw_negative_mass,
            empirical_cdf_at_quantiles=est.tolist(),
            standardized_cdf_errors=z.tolist(),
        )
        self.reference_audit.append(row)
        self.pivots[key] = p
        (self.out / "reference_audit.json").write_text(json.dumps(self.reference_audit, indent=2))
        return p

    def logrow(self, logR, pivot, base, kappa, valid=None):
        if valid is None:
            valid = np.isfinite(logR)
        lo = logR < pivot.quantiles[0]
        hi = logR > pivot.quantiles[2]
        central = pivot.central(logR) & valid
        delta = (np.abs(logR) <= Z975 * np.sqrt(kappa / base["n"])) & valid
        corr = (np.abs(logR - pivot.mean) <= Z975 * np.sqrt(pivot.variance)) & valid
        pit = pivot.prob(logR)
        pit = pit[valid]
        sp = np.sort(pit)
        R = len(sp)
        ks = (
            float(max(np.max(np.arange(1, R + 1) / R - sp), np.max(sp - np.arange(R) / R)))
            if R
            else np.nan
        )
        row = base | rates(central, valid) | summarize(np.sqrt(base["n"]) * logR, "scaled_log_")
        row.update(
            tail_lower=float((lo & valid).mean()),
            tail_upper=float((hi & valid).mean()),
            delta_coverage=float(delta.mean()),
            delta_mcse=float(np.sqrt(delta.mean() * (1 - delta.mean()) / len(logR))),
            corrected_normal_coverage=float(corr.mean()),
            pit_KS=ks,
        )
        row["transport_minus_delta"], row["paired_mcse"] = paired(central, delta)
        return row

    def block(self):
        start = time.time()
        rows = []
        precision = []
        samples = {}
        for r in (1, 3):
            for n in (50, 200, 1000):
                sn = self.pivot(n, list(n - np.arange(2 * r)), [1] * (2 * r))
                st = self.pivot(n, list(n - np.arange(r)) * 2, [1] * (2 * r))
                prec = self.pivot(n, [n - 2 * r + 1], [1]) if r == 3 else None
                vals = {rho: [] for rho in (0.0, 0.5, 0.9, 0.99)}
                pv = {rho: [] for rho in vals}
                rng = generator(self.seed, 200, r, n)
                batch = max(64, min(512, 1_000_000 // (n * 2 * r)))
                for offset in range(0, self.reps, batch):
                    b = min(batch, self.reps - offset)
                    obs = rng.standard_normal((b, n, 2 * r))
                    U = obs[:, :, :r]
                    V = obs[:, :, r:]
                    for rho in vals:
                        Y = np.concatenate((U, rho * U + np.sqrt(1 - rho * rho) * V), axis=2)
                        S = np.matmul(Y.transpose(0, 2, 1), Y) / n
                        A = S[:, :r, :r]
                        B = S[:, r:, r:]
                        C = S[:, :r, r:]
                        lsn = chol_logdet(S) - r * np.log1p(-rho * rho)
                        lst = chol_logdet(A) + chol_logdet(B)
                        overlap = np.einsum(
                            "bij,bji->b",
                            np.linalg.solve(A, C),
                            np.linalg.solve(B, C.transpose(0, 2, 1)),
                        )
                        kap = 4 * r + 4 * overlap
                        vals[rho].append(np.column_stack((lsn, lst, kap)))
                        if r == 3:
                            u = np.ones(2 * r) / np.sqrt(2 * r)
                            den = np.einsum(
                                "i,bi->b",
                                u,
                                np.linalg.solve(S, np.broadcast_to(u, (b, 2 * r))[..., None])[
                                    ..., 0
                                ],
                            )
                            # For equal canonical correlations and this u, theta=1+rho.
                            pv[rho].append(-np.log(den) - np.log1p(rho))
                for rho, blocks in vals.items():
                    arr = np.concatenate(blocks)
                    samples[f"r{r}_n{n}_rho{rho:g}"] = arr
                    base = dict(experiment="block", r=r, n=n, rho=rho, reps=self.reps)
                    rows.append(self.logrow(arr[:, 0], sn, base | dict(denominator="SN"), 4 * r))
                    rows.append(
                        self.logrow(arr[:, 1], st, base | dict(denominator="stable"), arr[:, 2])
                    )
                    if r == 3:
                        lp = np.concatenate(pv[rho])
                        precision.append(
                            self.logrow(
                                lp, prec, base | dict(denominator="precision_reciprocal"), 2
                            )
                        )
                print(f"block r={r}, n={n} complete", flush=True)
        pd.DataFrame(rows).to_csv(self.out / "block_results.csv", index=False)
        pd.DataFrame(precision).to_csv(self.out / "precision_results.csv", index=False)
        np.savez_compressed(self.out / "block_replicates.npz", **samples)
        self.timings["block"] = time.time() - start

    @staticmethod
    def standardized_summary(x):
        """OLS scores, computed from observed standardized rows, divisor n."""
        n = x.shape[1]
        S = np.matmul(x.transpose(0, 2, 1), x) / n
        U, V, W = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        suu = S[:, 0, 0]
        ga = S[:, 0, 1] / suu
        rm = V - ga[:, None] * U
        vm = np.mean(rm * rm, axis=1)
        gb = np.mean(rm * W, axis=1) / vm
        ew = W - (S[:, 0, 2] / suu)[:, None] * U - gb[:, None] * rm
        ve = np.mean(ew * ew, axis=1)
        sa = U * rm / suu[:, None]
        sb = rm * ew / vm[:, None]
        sa -= sa.mean(axis=1)[:, None]
        sb -= sb.mean(axis=1)[:, None]
        qa = np.mean(sa * sa, axis=1)
        qb = np.mean(sb * sb, axis=1)
        qab = np.mean(sa * sb, axis=1)
        logD = 2 * np.log(suu) + np.log(vm)
        sdlog = 2 * (U * U / suu[:, None] - 1) + (rm * rm / vm[:, None] - 1)
        vd = np.mean(sdlog**2, axis=1)
        return np.column_stack((ga, gb, qa, qb, qab, vm / suu, ve / vm, logD, vd))

    def generate_mediation(self, law, n, tag=300):
        rng = generator(self.seed, tag, {"G": 0, "E": 1, "D": 2}[law], n)
        result = []
        batch = max(64, min(512, 1_000_000 // (3 * n)))
        for offset in range(0, self.reps, batch):
            b = min(batch, self.reps - offset)
            if law == "E":
                x = rng.exponential(1, (b, n, 3)) - 1
            else:
                x = rng.standard_normal((b, n, 3))
                if law == "D":
                    x[:, :, 2] *= x[:, :, 1]
            result.append(self.standardized_summary(x))
        return np.concatenate(result)

    @staticmethod
    def effect_statistics(arr, n, a, b, v, sigma=np.sqrt(0.75)):
        ga, gb, qa, qb, qab, qag, qbg, logD, vd = arr.T
        sqv = np.sqrt(v)
        ah = a + sqv * ga
        bh = b + sigma / sqv * gb
        F = a * sigma * gb + b * v * ga + sigma * sqv * ga * gb
        ca = bh * v
        cb = sigma * ah
        ve = ca * ca * qa + cb * cb * qb + 2 * ca * cb * qab
        vg = ca * ca * qag + cb * cb * qbg
        with np.errstate(invalid="ignore", divide="ignore"):
            te = np.sqrt(n) * F / np.sqrt(ve)
            tg = np.sqrt(n) * F / np.sqrt(vg)
            se = np.sqrt(ve / (n * v))
            sg = np.sqrt(vg / (n * v))
        return dict(
            error=F / sqv,
            T_emp=te,
            T_G=tg,
            se_emp=se,
            se_G=sg,
            valid_emp=np.isfinite(te) & (ve > 0),
            valid_G=np.isfinite(tg) & (vg > 0),
        )

    def mediation(self):
        start = time.time()
        rows = []
        denrows = []
        allcache = {}
        sigma = np.sqrt(0.75)
        for law in ("G", "E", "D"):
            for n in (100, 300, 1000):
                arr = self.generate_mediation(law, n)
                allcache[f"{law}_n{n}"] = arr
                pivot = self.pivot(n, [n, n - 1], [2, 1])
                logD = arr[:, 7]
                br = dict(law=law, n=n, reps=self.reps)
                for m, mask in [
                    ("Gaussian_pivot", pivot.central(logD)),
                    ("Gaussian_log_delta", np.abs(logD) <= Z975 * np.sqrt(10 / n)),
                    ("Empirical_log_delta", np.abs(logD) <= Z975 * np.sqrt(arr[:, 8] / n)),
                ]:
                    denrows.append(
                        br
                        | dict(method=m)
                        | rates(mask, np.isfinite(logD))
                        | summarize(np.sqrt(n) * logD, "scaled_log_")
                    )
                for vname, v in [
                    ("1", 1.0),
                    ("n^-1/2", n**-0.5),
                    ("n^-1", 1 / n),
                    ("n^-2", n**-2.0),
                ]:
                    st = self.effect_statistics(arr, n, 0.8, 1.0, v)
                    me = (abs(st["T_emp"]) <= Z975) & st["valid_emp"]
                    mg = (abs(st["T_G"]) <= Z975) & st["valid_G"]
                    diff, ps = paired(me, mg)
                    wG = v * v + 0.8**2 * 0.75
                    lam = (v * v + 3 * 0.8**2 * 0.75) / wG if law == "D" else 1.0
                    for J in ("emp", "G"):
                        valid = st["valid_" + J]
                        T = st["T_" + J]
                        se = st["se_" + J]
                        row = br | dict(
                            v_name=vname,
                            v=v,
                            method=J,
                            variance_ratio=lam,
                            Gaussian_coverage_benchmark=float(
                                2 * norm.cdf(Z975 / np.sqrt(lam)) - 1
                            ),
                            emp_minus_G=diff,
                            paired_mcse=ps,
                            fieller_factor=float((1 - 10 * Z975**2 / n) ** -0.5),
                        )
                        row |= (
                            rates(abs(T) <= Z975, valid)
                            | summarize(T, "T_")
                            | summarize(se, "se_")
                            | summarize(st["error"], "error_")
                        )
                        f = (1 - 10 * Z975**2 / n) ** -0.5
                        row["enlarged_G_coverage"] = float(
                            ((abs(st["T_G"]) <= Z975 * f) & st["valid_G"]).mean()
                        )
                        rows.append(row)
                print(f"mediation law={law}, n={n} complete", flush=True)
        pd.DataFrame(rows).to_csv(self.out / "mediation_results.csv", index=False)
        pd.DataFrame(denrows).to_csv(self.out / "mediation_denominator_results.csv", index=False)
        np.savez_compressed(self.out / "mediation_replicates.npz", **allcache)
        self.timings["mediation"] = time.time() - start

    def canonical(self):
        start = time.time()
        n = 300
        arr = self.generate_mediation("G", n, tag=310)
        ga, gb, qa, qb, qab, qag, qbg, _, _ = arr.T
        z1 = np.sqrt(n) * ga
        z2 = np.sqrt(n) * gb
        rng = generator(self.seed, 311)
        lim = rng.standard_normal((self.limit_reps, 2))
        Z1, Z2 = lim.T
        rows = []
        for ha in HGRID:
            for hb in HGRID:
                x = ha + z1
                y = hb + z2
                H = ha * z2 + hb * z1 + z1 * z2
                with np.errstate(invalid="ignore", divide="ignore"):
                    te = H / np.sqrt(y * y * qa + x * x * qb + 2 * x * y * qab)
                    tg = H / np.sqrt(y * y * qag + x * x * qbg)
                    tl = (ha * Z2 + hb * Z1 + Z1 * Z2) / np.sqrt((ha + Z1) ** 2 + (hb + Z2) ** 2)
                lc = float((abs(tl) <= Z975).mean())
                for J, T in [("emp", te), ("G", tg)]:
                    row = dict(
                        n=n,
                        h_a=ha,
                        h_b=hb,
                        method=J,
                        reps=self.reps,
                        limit_coverage=lc,
                        limit_mcse=float(np.sqrt(lc * (1 - lc) / self.limit_reps)),
                        limit_reps=self.limit_reps,
                    )
                    row |= rates(abs(T) <= Z975, np.isfinite(T)) | summarize(T, "T_")
                    rows.append(row)
        pd.DataFrame(rows).to_csv(self.out / "canonical_results.csv", index=False)
        np.savez_compressed(self.out / "canonical_replicates.npz", statistics=arr, h_grid=HGRID)
        print("canonical plane complete", flush=True)
        self.timings["canonical"] = time.time() - start

    def adjusted(self):
        start = time.time()
        q = 4
        rows = []
        Gamma = 0.5 ** np.abs(np.subtract.outer(np.arange(q), np.arange(q)))
        L = np.linalg.cholesky(Gamma)
        popld = np.linalg.slogdet(Gamma)[1]
        aa = np.arange(1, q + 1) / 10
        bb = (-1.0) ** np.arange(q) * np.arange(1, q + 1) / 10
        for n in (50, 200, 1000):
            psn = self.pivot(n, list(n - np.arange(q)) + [n - q, n - q - 1], [2] * q + [1, 1])
            pst = self.pivot(n, list(n - np.arange(q)) + [n - q, n - q], [2] * q + [1, 1])
            vals = {(rho, v): [] for rho in (0.0, 0.5, 0.9, 0.99) for v in (1.0, 1e-6)}
            rng = generator(self.seed, 400, n)
            batch = max(64, min(512, 1_000_000 // (6 * n)))
            for offset in range(0, self.reps, batch):
                b = min(batch, self.reps - offset)
                zz = rng.standard_normal((b, n, 6))
                C = zz[:, :, :q] @ L.T
                U = zz[:, :, q]
                V = zz[:, :, q + 1]
                G = C.transpose(0, 2, 1) @ C / n
                ld = chol_logdet(G)
                for rho, v in vals:
                    x1 = C @ aa + U
                    x2 = C @ bb + np.sqrt(v) * (rho * U + np.sqrt(1 - rho * rho) * V)
                    XX = np.stack((x1, x2), axis=2)
                    coef = np.linalg.solve(G, C.transpose(0, 2, 1) @ XX / n)
                    resid = XX - C @ coef
                    H = resid.transpose(0, 2, 1) @ resid / n
                    lsn = 2 * (ld - popld) + chol_logdet(H) - np.log(v) - np.log1p(-rho * rho)
                    lst = 2 * (ld - popld) + np.log(H[:, 0, 0]) + np.log(H[:, 1, 1]) - np.log(v)
                    rh2 = H[:, 0, 1] ** 2 / (H[:, 0, 0] * H[:, 1, 1])
                    vals[(rho, v)].append(np.column_stack((lsn, lst, 8 * q + 4 + 4 * rh2)))
            for (rho, v), chunks in vals.items():
                ar = np.concatenate(chunks)
                base = dict(q=q, n=n, rho=rho, v2=v, reps=self.reps)
                rows.append(self.logrow(ar[:, 0], psn, base | dict(denominator="SN"), 8 * q + 4))
                rows.append(self.logrow(ar[:, 1], pst, base | dict(denominator="stable"), ar[:, 2]))
            print(f"adjusted q=4 n={n} complete", flush=True)
        pd.DataFrame(rows).to_csv(self.out / "adjusted_results.csv", index=False)
        self.timings["adjusted"] = time.time() - start

    def boundary(self):
        start = time.time()
        rows = []
        weak = []
        samples = {}
        for n in (50, 200, 1000, 5000):
            rng = generator(self.seed, 500, n)
            batch = max(32, min(512, 1_000_000 // (3 * n)))
            pieces = []
            wvals = {h: [] for h in (1.0, 2.0)}
            for offset in range(0, self.reps, batch):
                b = min(batch, self.reps - offset)
                x = rng.standard_normal((b, n, 3))
                S = x.transpose(0, 2, 1) @ x / n
                a = S[:, 0, 0]
                u = S[:, 0, 1]
                z = S[:, 0, 2]
                d = S[:, 1, 1]
                e = S[:, 1, 2]
                sqt = 1 / np.sqrt(n)
                t = 1 / n
                de = np.exp(chol_logdet(S[:, :2, :2]))
                full = np.exp(chol_logdet(S))
                ww = a + 2 * sqt * u + t * d
                r0 = de * full
                rn = (r0 + ww * (de + (a * e - u * z) / sqt) ** 2) / (2 + t)
                pieces.append(np.column_stack((r0, rn)))
                for h in wvals:
                    rho = h / np.sqrt(n)
                    cross = rho * a + np.sqrt(1 - rho * rho) * u
                    vy = rho * rho * a + 2 * rho * np.sqrt(1 - rho * rho) * u + (1 - rho * rho) * d
                    rel = cross / rho
                    FD = n * cross * cross / (a * vy + cross * cross)
                    wvals[h].append(np.column_stack((rel, FD)))
            ar = np.concatenate(pieces)
            samples[f"n{n}"] = ar
            for j, name in enumerate(("SN", "Napkin")):
                R = ar[:, j]
                mask = np.abs(R - 1) > 0.5
                row = dict(
                    n=n,
                    reps=self.reps,
                    denominator=name,
                    escape_probability=float(mask.mean()),
                    escape_lower=float(np.mean(R < 0.5)),
                    escape_upper=float(np.mean(R > 1.5)),
                    escape_mcse=float(np.sqrt(mask.mean() * (1 - mask.mean()) / self.reps)),
                    failures=int((~np.isfinite(R)).sum()),
                ) | summarize(R, "relative_")
                rows.append(row)
            for h, p in wvals.items():
                p = np.concatenate(p)
                mask = np.abs(p[:, 0] - 1) > 0.5
                weak.append(
                    dict(
                        n=n,
                        h=h,
                        reps=self.reps,
                        escape_probability=float(mask.mean()),
                        escape_mcse=float(np.sqrt(mask.mean() * (1 - mask.mean()) / self.reps)),
                        limit_escape=float(2 * norm.sf(h / 2)),
                        failures=int((~np.isfinite(p)).sum()),
                    )
                    | summarize(p[:, 0], "relative_")
                    | summarize(p[:, 1], "F_")
                )
            print(f"boundary n={n} complete", flush=True)
        pd.DataFrame(rows).to_csv(self.out / "napkin_results.csv", index=False)
        pd.DataFrame(weak).to_csv(self.out / "weak_results.csv", index=False)
        np.savez_compressed(self.out / "boundary_replicates.npz", **samples)
        self.timings["boundary"] = time.time() - start

    def run(self, stages):
        start = time.time()
        for stage in stages:
            getattr(self, stage)()
        benchmarks = {
            "transport_rho099": float(2 * norm.cdf(Z975 / np.sqrt(1 + 0.99**2)) - 1),
            "exponential_denominator": float(2 * norm.cdf(Z975 / 2) - 1),
            "dependent_effect_boundary": float(2 * norm.cdf(Z975 / np.sqrt(3)) - 1),
            "dependent_effect_interior_v1": float(2 * norm.cdf(Z975 / np.sqrt(2.44 / 1.48)) - 1),
            "napkin_escape": float(norm.sf(np.sqrt(2) - 1) + norm.cdf(-np.sqrt(2) - 1)),
            "double_null_coverage": float(2 * norm.cdf(2 * Z975) - 1),
        }
        (self.out / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2))
        manifest = dict(
            seed=self.seed,
            evaluation_repetitions=self.reps,
            reference_validation_repetitions=self.reference,
            canonical_limit_repetitions=self.limit_reps,
            stages=stages,
            stage_seconds=self.timings,
            total_seconds=time.time() - start,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pandas=pd.__version__,
            generator="numpy.random.default_rng; PCG64; SeedSequence([42, stage identifiers])",
            common_random_numbers="Within block r,n across rho; within mediation law,n across v; throughout canonical grid.",
            failure_policy="Undefined methods count as noncoverage; conditional coverage is also stored.",
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (self.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=50_000)
    parser.add_argument("--reference-reps", type=int, default=2_000_000)
    parser.add_argument("--limit-reps", type=int, default=1_000_000)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["block", "mediation", "canonical", "adjusted", "boundary"],
        default=["block", "mediation", "canonical", "adjusted", "boundary"],
    )
    args = parser.parse_args()
    if min(args.reps, args.reference_reps, args.limit_reps) < 50:
        parser.error("At least 50 repetitions in each role.")
    Study(args.out, args.seed, args.reps, args.reference_reps, args.limit_reps).run(args.stages)


if __name__ == "__main__":
    main()
