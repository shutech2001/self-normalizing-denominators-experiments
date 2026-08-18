from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import lstsq

DEFAULT_SEED = 42
RHC_URL = "https://hbiostat.org/data/repo/rhc.csv"
RHC_SHA256 = "9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8"
CONTINUOUS_COVARIATES = ("age", "surv2md1", "aps1")
CATEGORICAL_COVARIATES = ("sex", "cat1", "cat2", "dnr1")
TREATMENT_PROXIES = ("pafi1", "paco21")
OUTCOME_PROXIES = ("ph1", "hema1")
RESPONSE_COLUMNS = ("A", "Y", *TREATMENT_PROXIES, *OUTCOME_PROXIES)
PROXY_PAIRS = tuple(
    (treatment_proxy, outcome_proxy)
    for treatment_proxy in TREATMENT_PROXIES
    for outcome_proxy in OUTCOME_PROXIES
)


@dataclass(frozen=True)
class PreparedData:
    residuals: dict[str, NDArray[np.float64]]
    n: int
    design_rank: int
    design_columns: tuple[str, ...]

    @property
    def residual_df(self) -> int:
        return self.n - self.design_rank


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The SHA-256 hash of the file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_rhc(path: Path) -> Path:
    """Download the versioned public CSV atomically when it is absent.

    Args:
        path (Path): The path to the file.

    Raises:
        RuntimeError: If the downloaded file has an unexpected SHA-256.
        RuntimeError: If the file cannot be obtained.

    Returns:
        Path: The path to the downloaded file.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(RHC_URL, partial)
        observed_hash = _sha256(partial)
        if observed_hash != RHC_SHA256:
            raise RuntimeError(
                "Downloaded RHC file has an unexpected SHA-256: "
                f"{observed_hash} (expected {RHC_SHA256})"
            )
        partial.replace(path)
    except Exception as error:  # pragma: no cover - depends on network state
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not obtain the verified SUPPORT/RHC data from {RHC_URL}. "
            "Supply a local CSV with --rhc."
        ) from error
    return path


def _validate_columns(data: pd.DataFrame) -> None:
    """Validate the columns of the data.

    Args:
        data (pd.DataFrame): The data to validate.

    Raises:
        ValueError: If the data is missing required columns.
    """
    required = {
        "swang1",
        "t3d30",
        *CONTINUOUS_COVARIATES,
        *CATEGORICAL_COVARIATES,
        *TREATMENT_PROXIES,
        *OUTCOME_PROXIES,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"RHC file is missing required columns: {missing}")
    if len(data) <= len(required):
        raise ValueError("RHC data contain too few rows for the requested analysis")


def _encode_treatment(values: pd.Series) -> NDArray[np.float64]:
    """Encode the treatment values.

    Args:
        values (pd.Series): The treatment values.

    Raises:
        ValueError: If the treatment values are missing.
        ValueError: If the treatment values are not exactly "RHC" or "No RHC".

    Returns:
        NDArray[np.float64]: The encoded treatment values.
    """
    normalized = values.astype("string").str.strip().str.casefold()
    if normalized.isna().any():
        raise ValueError("swang1 contains missing treatment values")
    observed = set(normalized.unique())
    expected = {"rhc", "no rhc"}
    if observed != expected:
        raise ValueError(
            "swang1 must contain exactly the levels 'RHC' and 'No RHC'; "
            f"observed {sorted(observed)}"
        )
    encoded = normalized.map({"rhc": 1.0, "no rhc": 0.0}).to_numpy()
    return np.asarray(encoded, dtype=np.float64)


def _numeric_without_missing(data: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """Convert a column to a numeric array without missing values.

    Args:
        data (pd.DataFrame): The data.
        name (str): The name of the column.

    Raises:
        ValueError: If the column contains missing or non-numeric values.
        ValueError: If the column contains infinite values.

    Returns:
        NDArray[np.float64]: The numeric array.
    """
    values = pd.to_numeric(data[name], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} contains missing or non-numeric analysis values")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains infinite analysis values")
    return array


def prepare_support_data(data: pd.DataFrame) -> PreparedData:
    """Fit the stated preprocessing and jointly residualize all six variables.

    Args:
        data (pd.DataFrame): The data to prepare.

    Raises:
        ValueError: If the data is missing required columns.
        ValueError: If the data contains missing or non-numeric values.
        ValueError: If the data contains infinite values.
        ValueError: If the preprocessed adjustment design contains non-finite values.
        ValueError: If the adjustment design leaves no positive residual degrees of freedom.

    Returns:
        PreparedData: The prepared data.
    """
    _validate_columns(data)
    covariates = data[[*CONTINUOUS_COVARIATES, *CATEGORICAL_COVARIATES]].copy()
    for name in CONTINUOUS_COVARIATES:
        values = pd.to_numeric(covariates[name], errors="coerce")
        observed = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"Continuous covariate {name} contains infinite values")
        median = values.median()
        if not np.isfinite(median):
            raise ValueError(f"Continuous covariate {name} has no observed values")
        covariates[name] = values.fillna(median)
    for name in CATEGORICAL_COVARIATES:
        values = covariates[name].astype("object")
        mode = values.mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Categorical covariate {name} has no observed values")
        covariates[name] = values.fillna(mode.iloc[0]).astype(str)

    encoded = pd.get_dummies(
        covariates,
        columns=list(CATEGORICAL_COVARIATES),
        drop_first=True,
        dtype=float,
    )
    encoded = encoded.loc[:, encoded.nunique(dropna=False) > 1]
    matrix = encoded.to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales[scales == 0.0] = 1.0
    design = np.column_stack([np.ones(len(data)), (matrix - means) / scales])
    if not np.isfinite(design).all():
        raise ValueError("The preprocessed adjustment design contains non-finite values")

    response = np.column_stack(
        [
            _encode_treatment(data["swang1"]),
            _numeric_without_missing(data, "t3d30"),
            *(_numeric_without_missing(data, name) for name in TREATMENT_PROXIES),
            *(_numeric_without_missing(data, name) for name in OUTCOME_PROXIES),
        ]
    )
    coefficients, _, rank, _ = lstsq(design, response, check_finite=False, lapack_driver="gelsy")
    if int(rank) >= len(data):
        raise ValueError("The adjustment design leaves no positive residual degrees of freedom")
    residual_matrix = response - design @ coefficients
    residuals = {name: residual_matrix[:, index] for index, name in enumerate(RESPONSE_COLUMNS)}
    column_names = ("intercept", *(str(column) for column in encoded.columns))
    return PreparedData(residuals, len(data), int(rank), column_names)


def _moment(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the moment of two arrays.

    Args:
        left (np.ndarray): The left array.
        right (np.ndarray): The right array.

    Returns:
        float: The moment of the two arrays.
    """
    return float(left @ right / len(left))


def pair_metrics(
    prepared: PreparedData, treatment_proxy: str, outcome_proxy: str
) -> dict[str, float | int | str]:
    """Compute the metrics for a pair of treatment and outcome proxies.

    Args:
        prepared (PreparedData): The prepared data.
        treatment_proxy (str): The treatment proxy.
        outcome_proxy (str): The outcome proxy.

    Returns:
        dict[str, float | int | str]: The metrics.
    """
    residuals = prepared.residuals
    a, y = residuals["A"], residuals["Y"]
    z, w = residuals[treatment_proxy], residuals[outcome_proxy]
    saa, szz, sww = _moment(a, a), _moment(z, z), _moment(w, w)
    saz, saw, szw = _moment(a, z), _moment(a, w), _moment(z, w)
    say, szy = _moment(a, y), _moment(z, y)
    denominator = saa * szw - saz * saw
    m_az = saa * szz - saz**2
    m_aw = saa * sww - saw**2
    base_quantities = np.array([saa, szz, sww, m_az, m_aw, denominator])
    if (
        not np.isfinite(base_quantities).all()
        or np.any(base_quantities[:5] <= 0.0)
        or denominator == 0.0
    ):
        raise ValueError(f"Degenerate residual covariance for {treatment_proxy}/{outcome_proxy}")
    partial_rho_squared = np.clip(denominator**2 / (m_az * m_aw), 0.0, np.nextafter(1.0, 0.0))
    marginal_rho_squared = np.clip(saz**2 / (saa * szz), 0.0, np.nextafter(1.0, 0.0))
    degrees = prepared.residual_df

    gaussian_variance = m_az * m_aw + 3.0 * denominator**2
    scores = np.column_stack([a * a, z * z, w * w, a * z, a * w, z * w])
    centered_scores = scores - scores.mean(axis=0)
    sandwich_gamma = centered_scores.T @ centered_scores / prepared.n
    gradient = np.array([szw, 0.0, 0.0, -saw, -saz, saa])
    sandwich_variance = float(gradient @ sandwich_gamma @ gradient)
    if (
        not np.isfinite([gaussian_variance, sandwich_variance]).all()
        or gaussian_variance <= 0.0
        or sandwich_variance <= 0.0
    ):
        raise ValueError(f"Non-positive denominator variance for {treatment_proxy}/{outcome_proxy}")
    numerator = szw * say - szy * saw
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = numerator / denominator
        gaussian_f = degrees * denominator**2 / gaussian_variance
        sandwich_f = degrees * denominator**2 / sandwich_variance

    return {
        "n": prepared.n,
        "residual_df": degrees,
        "treatment_proxy": treatment_proxy,
        "outcome_proxy": outcome_proxy,
        "naive_F_A_on_Z": degrees * marginal_rho_squared / (1.0 - marginal_rho_squared),
        "partial_F_Z_to_W_given_A": degrees * partial_rho_squared / (1.0 - partial_rho_squared),
        "gaussian_F_D": gaussian_f,
        "sandwich_F_D": sandwich_f,
        "proximal_estimate": estimate,
        "partial_correlation": math.copysign(math.sqrt(float(partial_rho_squared)), denominator),
    }


def _point_diagnostics(prepared: PreparedData) -> pd.DataFrame:
    """Compute the point diagnostics for the prepared data.

    Args:
        prepared (PreparedData): The prepared data.

    Returns:
        pd.DataFrame: The point diagnostics.
    """
    return pd.DataFrame(
        [
            pair_metrics(prepared, treatment_proxy, outcome_proxy)
            for treatment_proxy, outcome_proxy in PROXY_PAIRS
        ]
    )


def _bootstrap_chunk(
    data: pd.DataFrame, replications: list[int], seed: int
) -> list[dict[str, float | int | str]]:
    """Bootstrap the point diagnostics.

    Args:
        data (pd.DataFrame): The data to bootstrap.
        replications (list[int]): The replications to bootstrap.
        seed (int): The seed for the random number generator.

    Returns:
        list[dict[str, float | int | str]]: The bootstrap results.
    """
    records: list[dict[str, float | int | str]] = []
    n = len(data)
    for replication in replications:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, replication]))
        sampled = data.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        prepared = prepare_support_data(sampled)
        for treatment_proxy, outcome_proxy in PROXY_PAIRS:
            metrics = pair_metrics(prepared, treatment_proxy, outcome_proxy)
            records.append(
                {
                    "replication": replication,
                    "treatment_proxy": treatment_proxy,
                    "outcome_proxy": outcome_proxy,
                    "sandwich_F_D": float(metrics["sandwich_F_D"]),
                    "proximal_estimate": float(metrics["proximal_estimate"]),
                    "design_rank": prepared.design_rank,
                }
            )
    return records


def _resolve_jobs(requested: int, task_count: int) -> int:
    """Resolve the number of jobs to run.

    Args:
        requested (int): The requested number of jobs.
        task_count (int): The number of tasks.

    Returns:
        int: The number of jobs to run.
    """
    if requested < -1:
        raise ValueError("jobs must be -1, 0, or a positive integer")
    if task_count <= 0:
        return 1
    available = max(os.cpu_count() or 1, 1)
    if requested > 0:
        return min(requested, task_count)
    return min(available if requested < 0 else min(available, 8), task_count)


def run_bootstrap(
    data: pd.DataFrame, replications: int, seed: int = DEFAULT_SEED, jobs: int = 0
) -> pd.DataFrame:
    """Run the bootstrap.

    Args:
        data (pd.DataFrame): The data to bootstrap.
        replications (int): The number of replications.
        seed (int): The seed for the random number generator.
        jobs (int): The number of jobs to run.

    Returns:
        pd.DataFrame: The bootstrap results.
    """
    columns = [
        "replication",
        "treatment_proxy",
        "outcome_proxy",
        "sandwich_F_D",
        "proximal_estimate",
        "design_rank",
    ]
    if replications < 0:
        raise ValueError("replications must be nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if replications == 0:
        return pd.DataFrame(columns=columns)
    worker_count = _resolve_jobs(jobs, replications)
    chunks = [
        chunk.astype(int).tolist()
        for chunk in np.array_split(np.arange(replications), worker_count)
        if len(chunk)
    ]
    if worker_count == 1:
        records = _bootstrap_chunk(data, chunks[0], seed)
    else:
        records = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_bootstrap_chunk, data, chunk, seed) for chunk in chunks]
            completed = 0
            for future in as_completed(futures):
                chunk_records = future.result()
                records.extend(chunk_records)
                completed += len(chunk_records) // len(PROXY_PAIRS)
                print(
                    f"Finished {completed}/{replications} bootstrap replications",
                    flush=True,
                )
    result = pd.DataFrame.from_records(records, columns=columns)
    return result.sort_values(["replication", "treatment_proxy", "outcome_proxy"]).reset_index(
        drop=True
    )


def _add_bootstrap_intervals(diagnostics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Add bootstrap intervals to the diagnostics.

    Args:
        diagnostics (pd.DataFrame): The diagnostics.
        bootstrap (pd.DataFrame): The bootstrap results.

    Returns:
        pd.DataFrame: The diagnostics with bootstrap intervals.
    """
    if bootstrap.empty:
        return diagnostics
    intervals = (
        bootstrap.groupby(["treatment_proxy", "outcome_proxy"], sort=False)
        .agg(
            sandwich_F_D_q025=("sandwich_F_D", lambda values: np.quantile(values, 0.025)),
            sandwich_F_D_q975=("sandwich_F_D", lambda values: np.quantile(values, 0.975)),
            estimate_q025=("proximal_estimate", lambda values: np.quantile(values, 0.025)),
            estimate_q975=("proximal_estimate", lambda values: np.quantile(values, 0.975)),
        )
        .reset_index()
    )
    return diagnostics.merge(intervals, on=["treatment_proxy", "outcome_proxy"], how="left")


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a pandas DataFrame to a list of JSON records.

    Args:
        frame (pd.DataFrame): The DataFrame to convert.

    Returns:
        list[dict[str, object]]: The JSON records.
    """
    return json.loads(frame.to_json(orient="records"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        argv (Sequence[str] | None): The command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_data"))
    parser.add_argument("--rhc", type=Path, default=None, help="Local rhc.csv path")
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Worker processes; 0 uses up to 8 cores and -1 uses all cores",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main function.

    Args:
        argv (Sequence[str] | None): The command line arguments.

    Returns:
        int: The exit code.
    """
    args = parse_args(argv)
    if args.bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rhc_path = args.rhc if args.rhc is not None else output_dir / "rhc.csv"
    rhc_path = download_rhc(rhc_path)
    data_hash = _sha256(rhc_path)
    if data_hash != RHC_SHA256:
        if args.rhc is None:
            raise ValueError(
                "The cached RHC file does not match the archived SHA-256. "
                "Replace it, or pass an intentional alternate file with --rhc."
            )
        print(
            f"Warning: local RHC SHA-256 differs from the archived version: {data_hash}",
            flush=True,
        )
    data = pd.read_csv(rhc_path)
    prepared = prepare_support_data(data)
    diagnostics = _point_diagnostics(prepared)
    workers = _resolve_jobs(args.jobs, args.bootstrap)
    print(
        f"Running SUPPORT audit: n={prepared.n}, bootstrap={args.bootstrap}, "
        f"seed={args.seed}, workers={workers}",
        flush=True,
    )
    bootstrap = run_bootstrap(data, args.bootstrap, args.seed, args.jobs)
    diagnostics = _add_bootstrap_intervals(diagnostics, bootstrap)
    diagnostics.to_csv(output_dir / "support_diagnostics.csv", index=False)
    bootstrap.to_csv(output_dir / "support_bootstrap.csv", index=False)

    payload = {
        "config": {
            "seed": args.seed,
            "bootstrap_replications": args.bootstrap,
            "workers": workers,
        },
        "data": {
            "url": RHC_URL,
            "path": str(rhc_path.resolve()),
            "sha256": data_hash,
            "rows": prepared.n,
            "design_rank": prepared.design_rank,
            "residual_df": prepared.residual_df,
            "design_columns": list(prepared.design_columns),
        },
        "diagnostic_convention": (
            "The manuscript-specific strength diagnostics use residual_df times "
            "rho_squared/(1-rho_squared); they are not regression-package finite-sample F tests."
        ),
        "diagnostics": _json_records(diagnostics),
    }
    with (output_dir / "real_data_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Outputs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
