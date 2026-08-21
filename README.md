# Experiments for "Self-normalizing denominators in rational causal estimation"

Reproducible code for the numerical experiments and real data experiments in "[**Self-normalizing denominators in rational causal estimation**](https://arxiv.org/abs/2608.20223)".
The two analyses are independent and use seed 42 by default.

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

Run the numerical study with the shell entry point:

```bash
./scripts/run_numerical_experiments.sh
```

The shell script can be called from any directory and forwards all arguments to `experiments/numerical_experiments.py`. With no arguments, it runs the full experiment used for the manuscript:

- sample size: 1000;
- 100,000 Monte Carlo replications in each of four front-door and five
  proximal cells;
- batch size: 5,000;
- random seed: 42;
- up to eight worker processes; and
- output directory: `outputs/numerical`.

The simulation draws directly with `scipy.stats.wishart`.
Because the Gaussian mean is known to be zero, this is exactly the distribution of the sample covariance obtained by simulating all observations;
it is not an approximation.

For example, a short single-process check can be run with:

```bash
./scripts/run_numerical_experiments.sh \
  --replications 2000 --jobs 1 --output-dir outputs/numerical_quick
```

Useful command-line options are `--n`, `--replications`, `--batch-size`, `--seed`, `--jobs`, and `--output-dir`. `--jobs 0` uses up to eight workers, `--jobs -1` uses all available CPUs, and a positive value requests that many workers. Random-number streams are cell-specific, so changing `--jobs` does not change the results. Exact reproduction also requires the same batch size.

The full run writes the following files:

- `outputs/numerical/frontdoor_simulation.csv`: four cell-level front-door summaries;
- `outputs/numerical/proximal_simulation.csv`: five cell-level proximal summaries; and
- `outputs/numerical/numerical_results.json`: the complete configuration and both sets of summaries in JSON format.

The summaries include population and median $F_D$, proximal partial and naive $F$ diagnostics, robust standard deviation (IQR/1.349), median standard error and bias, Wald and inverted-moment coverage, and Monte Carlo standard errors for both coverage estimates. Running the default command again overwrites these generated CSV and JSON files;
use a different `--output-dir` for exploratory runs.

### Executing Real Data Experiments

Run the SUPPORT/RHC denominator audit with:

```bash
./scripts/run_real_data_experiments.sh
```

The shell script forwards all arguments to `experiments/real_data_experiments.py`.
Its default full run uses 2,000 row-bootstrap replications, random seed 42, up to eight workers, and `outputs/real_data` as the output directory.
If a local data file is not supplied and `outputs/real_data/rhc.csv` does not exist, the script downloads the public Vanderbilt SUPPORT/RHC CSV and verifies its SHA-256.

To use an existing copy of the data and perform a short single-process check:

```bash
./scripts/run_real_data_experiments.sh \
  --rhc /path/to/rhc.csv --bootstrap 200 --jobs 1 \
  --output-dir outputs/real_data_quick
```

Useful options are `--rhc`, `--bootstrap`, `--seed`, `--jobs`, and `--output-dir`;
`--jobs` follows the same convention as in the numerical experiment.

The source URL is <https://hbiostat.org/data/repo/rhc.csv>.
The archived file used to verify this implementation has 5,735 rows and SHA-256 `9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8`.
Locally supplied files are schema- and treatment-level-validated;
a checksum difference is reported but allowed.

Every bootstrap replication resamples patient rows and then refits median/mode imputation, one-hot encoding, and the joint residualization.
The full run writes:

- `outputs/real_data/rhc.csv`: the verified source data when downloaded by the script;
- `outputs/real_data/support_diagnostics.csv`: one row for each of the four proxy pairs, including point diagnostics and percentile intervals;
- `outputs/real_data/support_bootstrap.csv`: one row per bootstrap replication and proxy pair; and
- `outputs/real_data/real_data_results.json`: configuration, data provenance, design information, and point diagnostics in JSON format.

The analysis is a descriptive denominator audit.
The reported strength diagnostics and bootstrap percentiles do not establish proxy validity or a clinical causal effect.
As above, use a separate `--output-dir` when results from the existing full run should be retained.

## Citation
```bibtex
@article{tamano2026self,
    author={Tamano, Shu},
    journal={arXiv preprint arXiv:2608.20223},
    title={Self-normalizing denominators in rational causal estimation},
    year={2026},
}
```

## Contact
If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
