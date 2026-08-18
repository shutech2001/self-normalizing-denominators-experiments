from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2, iqr, norm, wishart

DEFAULT_SEED = 42
Z_975 = float(norm.ppf(0.975))
CHI2_95 = float(chi2.ppf(0.95, df=1))
FRONTDOOR_GRID = (1.0, 0.1, 0.01, 0.001)
PROXIMAL_GRID = (1.0, 0.3, 0.1, 0.03, 0.01)


@dataclass(frozen=True)
class SimulationConfig:
    n: int = 1_000
    replications: int = 100_000
    batch_size: int = 5_000
    seed: int = DEFAULT_SEED


def frontdoor_covariance(v_m: float) -> NDArray[np.float64]:
    """Covariance of (X, M, Y) in the paper's front-door design.

    Args:
        v_m (float): The variance of the mediator M.

    Returns:
        NDArray[np.float64]: The covariance matrix of (X, M, Y).
    """
    loading = np.array([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.8, 1.0, 1.0]])
    shock_covariance = np.array([[1.0, 0.0, 0.5], [0.0, v_m, 0.0], [0.5, 0.0, 1.0]])
    return loading @ shock_covariance @ loading.T


def proximal_covariance(v_a: float) -> NDArray[np.float64]:
    """Covariance of (Z, A, W, Y) in the paper's proximal design.

    Args:
        v_a (float): The variance of the treatment A.

    Returns:
        NDArray[np.float64]: The covariance matrix of (Z, A, W, Y).
    """
    loading = np.array(
        [
            [0.8, 1.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 1.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 1.0, 0.0],
            [1.6, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    shock_covariance = np.diag([1.0, 1.0, v_a, 1.0, 1.0])
    return loading @ shock_covariance @ loading.T


def _vech_indices(dimension: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Indices of the upper triangular part of a symmetric matrix.

    Args:
        dimension (int): The dimension of the symmetric matrix.

    Returns:
        tuple[NDArray[np.int32], NDArray[np.int32]]:
        The indices of the upper triangular part of the symmetric matrix.
    """
    pairs = [(i, j) for i in range(dimension) for j in range(i, dimension)]
    return (
        np.asarray([i for i, _ in pairs], dtype=np.int32),
        np.asarray([j for _, j in pairs], dtype=np.int32),
    )


def _gaussian_gamma(covariances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Covariance of sqrt(n) vech(S_hat - Sigma), evaluated by row.

    Args:
        covariances (NDArray[np.float64]): The sample covariance matrix.

    Returns:
        NDArray[np.float64]: The covariance of sqrt(n) vech(S_hat - Sigma), evaluated by row.
    """
    first, second = _vech_indices(covariances.shape[1])
    return (
        covariances[:, first[:, None], first[None, :]]
        * covariances[:, second[:, None], second[None, :]]
        + covariances[:, first[:, None], second[None, :]]
        * covariances[:, second[:, None], first[None, :]]
    )


def _frontdoor_polynomials(
    covariance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Polynomials for the front-door design.

    Args:
        covariance (NDArray[np.float64]): The covariance matrix.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        The polynomials.
    """
    x, xm, xy = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2]
    mm, my = covariance[:, 1, 1], covariance[:, 1, 2]
    numerator = xm * (x * my - xm * xy)
    denominator = x * (x * mm - xm**2)
    gradient_n = np.column_stack(
        [xm * my, x * my - 2.0 * xm * xy, -(xm**2), 0.0 * x, x * xm, 0.0 * x]
    )
    gradient_d = np.column_stack(
        [2.0 * x * mm - xm**2, -2.0 * x * xm, 0.0 * x, x**2, 0.0 * x, 0.0 * x]
    )
    return numerator, denominator, gradient_n, gradient_d


def _proximal_polynomials(
    covariance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Polynomials for the proximal design.

    Args:
        covariance (NDArray[np.float64]): The covariance matrix.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        The polynomials.
    """
    za = covariance[:, 0, 1]
    zw = covariance[:, 0, 2]
    zy = covariance[:, 0, 3]
    aa = covariance[:, 1, 1]
    aw = covariance[:, 1, 2]
    ay = covariance[:, 1, 3]
    numerator = zw * ay - zy * aw
    denominator = zw * aa - za * aw
    zeros = np.zeros_like(za)
    gradient_n = np.column_stack([zeros, zeros, ay, -aw, zeros, -zy, zw, zeros, zeros, zeros])
    gradient_d = np.column_stack([zeros, -aw, aa, zeros, zw, -za, zeros, zeros, zeros, zeros])
    return numerator, denominator, gradient_n, gradient_d


def _quadratic_form(
    vectors: NDArray[np.float64], matrices: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Quadratic form of a vector and a matrix.

    Args:
        vectors (NDArray[np.float64]): The vector.
        matrices (NDArray[np.float64]): The matrix.

    Returns:
        NDArray[np.float64]: The quadratic form of the vector and the matrix.
    """
    return np.einsum("bi,bij,bj->b", vectors, matrices, vectors, optimize=True)


def _ratio_metrics(
    covariances: NDArray[np.float64], family: str, sample_size: int
) -> dict[str, NDArray[np.float64]]:
    """Metrics for the front-door and proximal designs.

    Args:
        covariances (NDArray[np.float64]): The covariance matrix.
        family (str): The family of the design.
        sample_size (int): The sample size.

    Returns:
        dict[str, NDArray[np.float64]]: The metrics.
    """
    if family == "frontdoor":
        numerator, denominator, gradient_n, gradient_d = _frontdoor_polynomials(covariances)
        truth = 0.8
    elif family == "proximal":
        numerator, denominator, gradient_n, gradient_d = _proximal_polynomials(covariances)
        truth = 1.0
    else:  # pragma: no cover - guarded by the cell specification
        raise ValueError(f"Unknown experiment family: {family}")

    gamma = _gaussian_gamma(covariances)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        estimate = numerator / denominator
        influence = (gradient_n - estimate[:, None] * gradient_d) / denominator[:, None]
        variance = _quadratic_form(influence, gamma) / sample_size
        standard_error = np.sqrt(np.maximum(variance, 0.0))
        denominator_variance = _quadratic_form(gradient_d, gamma)
        f_d = sample_size * denominator**2 / denominator_variance
        null_gradient = gradient_n - truth * gradient_d
        null_variance = _quadratic_form(null_gradient, gamma)
        inversion_statistic = sample_size * (numerator - truth * denominator) ** 2 / null_variance
    if family == "frontdoor":
        # Algebraically exact for every positive-definite sample covariance.
        f_d.fill(sample_size / 10.0)

    valid = (
        np.isfinite(estimate)
        & np.isfinite(standard_error)
        & np.isfinite(f_d)
        & np.isfinite(inversion_statistic)
        & (variance >= 0.0)
        & (denominator_variance > 0.0)
        & (null_variance > 0.0)
    )
    result = {
        "estimate": estimate,
        "standard_error": standard_error,
        "F_D": f_d,
        "wald_covered": valid & (np.abs(estimate - truth) <= Z_975 * standard_error),
        "inversion_covered": valid & (inversion_statistic <= CHI2_95),
        "valid": valid,
    }

    if family == "proximal":
        zz, ww = covariances[:, 0, 0], covariances[:, 2, 2]
        rho_az_squared = covariances[:, 0, 1] ** 2 / (zz * covariances[:, 1, 1])
        m_az = covariances[:, 1, 1] * zz - covariances[:, 0, 1] ** 2
        m_aw = covariances[:, 1, 1] * ww - covariances[:, 1, 2] ** 2
        partial_rho_squared = denominator**2 / (m_az * m_aw)
        upper = np.nextafter(1.0, 0.0)
        rho_az_squared = np.clip(rho_az_squared, 0.0, upper)
        partial_rho_squared = np.clip(partial_rho_squared, 0.0, upper)
        result["naive_F"] = sample_size * rho_az_squared / (1.0 - rho_az_squared)
        result["partial_F"] = sample_size * partial_rho_squared / (1.0 - partial_rho_squared)
    return result


def _population_f(covariance: NDArray[np.float64], family: str, sample_size: int) -> float:
    """Population F_D for the front-door and proximal designs.

    Args:
        covariance (NDArray[np.float64]): The covariance matrix.
        family (str): The family of the design.
        sample_size (int): The sample size.

    Returns:
        float: The population F_D.
    """
    metrics = _ratio_metrics(covariance[None, :, :], family, sample_size)
    return float(metrics["F_D"][0])


def _summarize_cell(
    family: str, parameter: float, config: SimulationConfig
) -> dict[str, float | int | str]:
    """Summarize a cell of the front-door and proximal designs.

    Args:
        family (str): The family of the design.
        parameter (float): The parameter value.
        config (SimulationConfig): The configuration.

    Returns:
        dict[str, float | int | str]: The summary.
    """
    covariance = (
        frontdoor_covariance(parameter) if family == "frontdoor" else proximal_covariance(parameter)
    )
    family_code = 1 if family == "frontdoor" else 2
    cell_grid = FRONTDOOR_GRID if family == "frontdoor" else PROXIMAL_GRID
    cell_index = cell_grid.index(parameter)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, family_code, cell_index]))
    collected: dict[str, list[NDArray[np.float64]]] = {}
    remaining = config.replications
    while remaining:
        count = min(config.batch_size, remaining)
        draws = np.asarray(
            wishart.rvs(df=config.n, scale=covariance, size=count, random_state=rng),
            dtype=float,
        ).reshape(count, covariance.shape[0], covariance.shape[1])
        metrics = _ratio_metrics(draws / config.n, family, config.n)
        for name, vals in metrics.items():
            collected.setdefault(name, []).append(vals)
        remaining -= count

    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    valid = values["valid"]
    estimates = values["estimate"][valid]
    truth = 0.8 if family == "frontdoor" else 1.0
    if not estimates.size:
        raise RuntimeError(f"No valid replications for {family}, parameter={parameter}")

    wald_coverage = float(values["wald_covered"].sum() / valid.sum())
    inversion_coverage = float(values["inversion_covered"].sum() / valid.sum())
    row: dict[str, float | int | str] = {
        "experiment": family,
        "parameter": "v_M" if family == "frontdoor" else "v_A",
        "parameter_value": parameter,
        "n": config.n,
        "replications": config.replications,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "population_F_D": _population_f(covariance, family, config.n),
        "median_F_D": float(np.median(values["F_D"][valid])),
        "robust_sd": float(iqr(estimates) / 1.349),
        "median_se": float(np.median(values["standard_error"][valid])),
        "median_bias": float(np.median(estimates) - truth),
        "wald_coverage": wald_coverage,
        "wald_coverage_mcse": math.sqrt(wald_coverage * (1.0 - wald_coverage) / valid.sum()),
        "inversion_coverage": inversion_coverage,
        "inversion_coverage_mcse": math.sqrt(
            inversion_coverage * (1.0 - inversion_coverage) / valid.sum()
        ),
        "invalid_replications": int(config.replications - valid.sum()),
    }
    if family == "proximal":
        row["median_partial_F"] = float(np.median(values["partial_F"][valid]))
        row["median_naive_F"] = float(np.median(values["naive_F"][valid]))
    else:
        row["median_partial_F"] = math.nan
        row["median_naive_F"] = math.nan
    return row


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
    if requested > 0:
        return min(requested, task_count)
    available = max(os.cpu_count() or 1, 1)
    return min(available if requested < 0 else min(available, 8), task_count)


def run_experiments(config: SimulationConfig, jobs: int = 0) -> pd.DataFrame:
    """Run the experiments.

    Args:
        config (SimulationConfig): The configuration.
        jobs (int): The number of jobs to run.

    Returns:
        pd.DataFrame: The results.
    """
    cells = [
        *(("frontdoor", value) for value in FRONTDOOR_GRID),
        *(("proximal", value) for value in PROXIMAL_GRID),
    ]
    worker_count = _resolve_jobs(jobs, len(cells))
    if worker_count == 1:
        rows = [_summarize_cell(family, value, config) for family, value in cells]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_summarize_cell, family, value, config): (family, value)
                for family, value in cells
            }
            for future in as_completed(futures):
                family, value = futures[future]
                rows.append(future.result())
                print(f"Finished {family} cell at {value:g}", flush=True)
    order = {(family, value): i for i, (family, value) in enumerate(cells)}
    rows.sort(key=lambda row: order[(str(row["experiment"]), float(row["parameter_value"]))])
    return pd.DataFrame(rows)


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/numerical"))
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--replications", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
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
    if args.n <= 4 or args.replications <= 0 or args.batch_size <= 0:
        raise ValueError("n, replications, and batch-size must be positive (n > 4)")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    config = SimulationConfig(args.n, args.replications, args.batch_size, args.seed)
    worker_count = _resolve_jobs(args.jobs, len(FRONTDOOR_GRID) + len(PROXIMAL_GRID))
    print(
        f"Running numerical experiments: n={config.n}, B={config.replications}, "
        f"seed={config.seed}, workers={worker_count}",
        flush=True,
    )
    results = run_experiments(config, args.jobs)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frontdoor = results.loc[results["experiment"] == "frontdoor"].copy()
    proximal = results.loc[results["experiment"] == "proximal"].copy()
    frontdoor.to_csv(output_dir / "frontdoor_simulation.csv", index=False)
    proximal.to_csv(output_dir / "proximal_simulation.csv", index=False)
    payload = {
        "config": {
            **config.__dict__,
            "workers": worker_count,
            "normal_quantile_0.975": Z_975,
            "chi_squared_1_quantile_0.95": CHI2_95,
        },
        "frontdoor": _json_records(frontdoor),
        "proximal": _json_records(proximal),
    }
    with (output_dir / "numerical_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    display_columns = [
        "experiment",
        "parameter_value",
        "median_F_D",
        "median_partial_F",
        "median_naive_F",
        "robust_sd",
        "median_bias",
        "wald_coverage",
        "inversion_coverage",
    ]
    print(results[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Outputs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
