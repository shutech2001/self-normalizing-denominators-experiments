# Experiments for "Self-normalizing denominators in rational covariance estimators"

Reproducible code for the numerical experiments in "[**Self-normalizing denominators in rational covariance estimators**](https://arxiv.org/abs/2608.20223)".
The simulations use seed 42 by default.

## Reproducing the Experiments

### Requirements and Setup
```bash
# clone the repository
git clone git@github.com:shutech2001/self-normalizing-denominators-experiments.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

### Executing Numerical Experiments

Run the complete numerical study with the shell entry point:

```bash
./scripts/run_numerical_experiments.sh
```

The shell script can be called from any directory. It runs `experiments/simulate.py`, then `experiments/verify.py`, then `experiments/plot.py`, using Poetry from the repository root. If any command fails, later commands are not run. The simulation includes five stages:

1. `block`: block covariance denominators, calibration, and precision;
2. `mediation`: mediation effects and denominators under Gaussian, exponential, and dependent observations;
3. `canonical`: finite-sample Wald coverage over the canonical parameter grid and its limiting comparison;
4. `adjusted`: covariate-adjusted denominators; and
5. `boundary`: Napkin and weak-denominator boundary experiments.

The defaults are 50,000 evaluation repetitions (`--reps`), 2,000,000 reference validation repetitions (`--reference-reps`), 1,000,000 canonical limit repetitions (`--limit-reps`), and random seed 42 (`--seed`). Results are written to `outputs/` and figures to `figures/`.

For a reduced-repetition run in separate directories:

```bash
./scripts/run_numerical_experiments.sh \
  --reps 500 --reference-reps 2000 --limit-reps 2000 \
  --out outputs/quick --figure-dir figures/quick
```

Each repetition count must be at least 50. The reduced run checks the workflow; its Monte Carlo precision differs from the default study. Both `--out DIR` and `--out=DIR` are supported, as are both forms of the other options. Relative paths are interpreted from the repository root. Use `--help` to display options without starting a run.

The shell script always runs all five stages and then creates the full figure set, so it does not accept `--stages`. To run selected simulations, invoke `poetry run python experiments/simulate.py --stages block --out outputs/block_only` directly. To verify or redraw an existing complete run, use `poetry run python experiments/verify.py --out outputs` or `poetry run python experiments/plot.py --input-dir outputs --figure-dir figures`.

A complete run writes the following files under the selected output directory:

- CSV summaries: `block_results.csv`, `precision_results.csv`, `mediation_results.csv`, `mediation_denominator_results.csv`, `canonical_results.csv`, `adjusted_results.csv`, `napkin_results.csv`, and `weak_results.csv`;
- compressed replicate archives: `block_replicates.npz`, `mediation_replicates.npz`, `canonical_replicates.npz`, and `boundary_replicates.npz`;
- `reference_audit.json`: numerical reference-law diagnostics and Monte Carlo validation;
- `benchmarks.json`: theoretical benchmark probabilities;
- `run_manifest.json`: repetition counts, seed, stages, timings, software versions, and simulation source hash; and
- `validation_checks.json`: verification results for analytic identities, numerical accuracy, and random-number streams.

Plotting writes seven PDF files under the selected figure directory: `1a_quantiles.pdf`, `1b_calibration.pdf`, `2a_precision.pdf`, `2b_canonical.pdf`, `log_corrections.pdf`, `napkin.pdf`, and `napkin_qq.pdf`. PNG files are not generated. Reusing output or figure directories overwrites files with the same names; select different directories for exploratory runs.

## Citation
```bibtex
@article{tamano2026self,
    author={Tamano, Shu},
    journal={arXiv preprint arXiv:2608.20223},
    title={Self-normalizing denominators in rational covariance estimators},
    year={2026},
}
```

## Contact
If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
