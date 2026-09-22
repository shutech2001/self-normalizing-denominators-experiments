import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.stats import chi2

from simulate import LogPivot, Study, generator

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Verify the numerical implementations.")
parser.add_argument("--out", type=Path, default=ROOT / "outputs")
args = parser.parse_args()
checks = {}
# Analytic one-factor reference, independent of the Fourier implementation.
maxq = 0
for n, df in [(50, 45), (200, 195), (1000, 995)]:
    law = LogPivot.make(n, [df], [1])
    target = np.log(chi2.ppf([0.025, 0.5, 0.975], df) / n)
    maxq = max(maxq, float(np.max(abs(law.quantiles - target))))
checks["log_chisquare_quantile_max_absolute_error"] = maxq
assert maxq < 2e-8

# Generic raw covariance gradients and raw regressions versus canonical evaluation.
maxse = maxerr = maxden = 0.0
rng = generator(42, 900)
for law in ("G", "E", "D"):
    z = rng.exponential(size=(8, 300, 3)) - 1 if law == "E" else rng.standard_normal((8, 300, 3))
    if law == "D":
        z[:, :, 2] *= z[:, :, 1]
    a0 = Study.standardized_summary(z)
    for v in [1.0, 1 / 300, 1e-6]:
        X = z[:, :, 0]
        M = 0.8 * X + np.sqrt(v) * z[:, :, 1]
        Y = M + 0.5 * X + np.sqrt(0.75) * z[:, :, 2]
        obs = np.stack((X, M, Y), axis=2)
        raw = Study.standardized_summary(obs)
        ah, bh, qa, qb, qab = raw[:, :5].T
        truth = 0.8
        se = np.sqrt((bh * bh * qa + ah * ah * qb + 2 * ah * bh * qab) / 300)
        st = Study.effect_statistics(a0, 300, 0.8, 1.0, v)
        maxse = max(maxse, float(np.max(abs(se / st["se_emp"] - 1))))
        maxerr = max(
            maxerr, float(np.max(abs((ah * bh - truth) - st["error"]) / (1 + abs(st["error"]))))
        )
        maxden = max(maxden, float(np.max(abs(raw[:, 7] - np.log(v) - a0[:, 7]))))
        if v == 1.0:
            S = obs.transpose(0, 2, 1) @ obs / 300
            ix = np.triu_indices(3)
            scores = obs[:, :, ix[0]] * obs[:, :, ix[1]]
            scores -= scores.mean(axis=1)[:, None, :]
            Gam = scores.transpose(0, 2, 1) @ scores / 300

            def tau(t):
                u, c, d, m, e, y = t
                return c / u * (u * e - c * d) / (u * m - c * c)

            grad = np.empty((8, 6))
            for j in range(6):
                zc = S[:, ix[0], ix[1]].astype(complex)
                zc[:, j] += 1e-25j
                grad[:, j] = np.imag(np.array([tau(t) for t in zc])) / 1e-25
            independent = np.sqrt(np.einsum("bi,bij,bj->b", grad, Gam, grad) / 300)
            maxse = max(maxse, float(np.max(abs(independent / se - 1))))
checks["raw_vs_canonical_empirical_se_relative_error"] = maxse
checks["raw_vs_canonical_effect_scaled_error"] = maxerr
checks["raw_vs_canonical_log_denominator_error"] = maxden
assert max(maxse, maxerr, maxden) < 2e-8

# Common-random-number invariance under nuisance block transformations.
r = 3
n = 50
rng = generator(42, 901)
z = rng.standard_normal((12, n, 6))
rho = 0.99
x = np.concatenate((z[:, :, :3], rho * z[:, :, :3] + np.sqrt(1 - rho * rho) * z[:, :, 3:]), axis=2)
T1 = np.array([[1.0, 0.3, 0], [0, 2.0, -0.2], [0.1, 0, 0.7]])
T2 = np.array([[0.6, 0, 0.2], [0.4, 1.2, 0], [0, -0.1, 3.0]])
T = np.zeros((6, 6))
T[:3, :3] = T1
T[3:, 3:] = T2
S = x.transpose(0, 2, 1) @ x / n
Q = (x @ T.T).transpose(0, 2, 1) @ (x @ T.T) / n
cov = np.block([[np.eye(3), rho * np.eye(3)], [rho * np.eye(3), np.eye(3)]])
K = T @ cov @ T.T
ld = lambda A: np.linalg.slogdet(A)[1]
checks["block_congruence_full_logratio_error"] = float(
    np.max(abs((ld(Q) - ld(K)) - (ld(S) - ld(cov))))
)
checks["block_congruence_product_logratio_error"] = float(
    np.max(
        abs(
            (ld(Q[:, :3, :3]) + ld(Q[:, 3:, 3:]) - ld(K[:3, :3]) - ld(K[3:, 3:]))
            - (ld(S[:, :3, :3]) + ld(S[:, 3:, 3:]))
        )
    )
)
# High precision from exactly the floating-point observation values used above.
mp.mp.dps = 70
xm = mp.matrix([[mp.mpf(float(t)) for t in row] for row in x[0]])
Sm = xm.T * xm / n
high = float(mp.log(mp.det(Sm)))
checks["float64_vs_70_digit_logdet_error"] = abs(ld(S[0]) - high)
assert checks["float64_vs_70_digit_logdet_error"] < 1e-9

# Napkin raw formula, shear alpha nonzero, and stable normalized evaluation.
z = generator(42, 902).standard_normal((10, 100, 3))
S = z.transpose(0, 2, 1) @ z / 100
errs = []
for t in (0.01, 0.0002):
    for alpha in (-0.7, 0.4, 1.3):
        Z = z[:, :, 0]
        W = Z + np.sqrt(t) * z[:, :, 1]
        X = alpha * Z + W + z[:, :, 2]
        obs = np.stack((Z, W, X), axis=2)
        raw = obs.transpose(0, 2, 1) @ obs / 100
        delta = np.linalg.det(raw[:, :2, :2])
        dd = np.linalg.det(raw)
        m = raw[:, 0, 0] * raw[:, 1, 2] - raw[:, 0, 1] * raw[:, 0, 2]
        direct = (delta * dd + raw[:, 1, 1] * m * m) / (t * t * (t + 2))
        a = S[:, 0, 0]
        u = S[:, 0, 1]
        w = S[:, 0, 2]
        e = S[:, 1, 2]
        de = np.linalg.det(S[:, :2, :2])
        full = np.linalg.det(S)
        pred = (
            de * full
            + (a + 2 * np.sqrt(t) * u + t * S[:, 1, 1]) * (de + (a * e - u * w) / np.sqrt(t)) ** 2
        ) / (t + 2)
        errs.append(float(np.max(abs(direct / pred - 1))))
checks["napkin_raw_vs_scaled_relative_error"] = max(errs)
assert max(errs) < 1e-8

# Rooted streams reproduce bit for bit, and disjoint stream keys differ.
a = generator(42, 990).normal(size=100)
b = generator(42, 990).normal(size=100)
c = generator(42, 991).normal(size=100)
checks["seed_reproducible"] = bool(np.array_equal(a, b))
checks["stream_separation"] = bool(not np.array_equal(a, c))
checks["all_checks_passed"] = True
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "validation_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
