#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_DIR"

usage() {
    cat <<'EOF'
Usage: ./scripts/run_numerical_experiments.sh [options]

Run all five simulation stages (block, mediation, canonical, adjusted, boundary),
verify the numerical identities, and generate seven PDF figures.

Options:
  --out DIR             Simulation and validation output (default: outputs)
  --figure-dir DIR      PDF figure output (default: figures)
  --reps N              Evaluation repetitions (default: 50000)
  --reference-reps N    Reference validation repetitions (default: 2000000)
  --limit-reps N        Canonical limit repetitions (default: 1000000)
  --seed N              Random seed (default: 42)
  -h, --help            Show this help without running experiments

Options accept both --option VALUE and --option=VALUE. Relative directories
are resolved from the repository root. Each repetition count must be at least 50.
For individual stages, use poetry run python experiments/simulate.py --stages ...
directly; this complete pipeline does not accept --stages.
EOF
}

fail() {
    printf '%s\n' "$1" >&2
    exit 2
}

output_dir="$REPOSITORY_DIR/outputs"
figure_dir="$REPOSITORY_DIR/figures"
reps=50000
reference_reps=2000000
limit_reps=1000000
seed=42

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --stages|--stages=*)
            fail "--stages is unavailable here: this pipeline runs the complete study. Run poetry run python experiments/simulate.py --stages ... directly for a partial study."
            ;;
        --out|--figure-dir|--reps|--reference-reps|--limit-reps|--seed)
            option="$1"
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $option."
            value="$2"
            shift 2
            ;;
        --out=*|--figure-dir=*|--reps=*|--reference-reps=*|--limit-reps=*|--seed=*)
            option="${1%%=*}"
            value="${1#*=}"
            [[ -n "$value" ]] || fail "Missing value for $option."
            shift
            ;;
        *)
            fail "Unknown option: $1. Use --help for supported options."
            ;;
    esac
    case "$option" in
        --out) output_dir="$value" ;;
        --figure-dir) figure_dir="$value" ;;
        --reps) reps="$value" ;;
        --reference-reps) reference_reps="$value" ;;
        --limit-reps) limit_reps="$value" ;;
        --seed) seed="$value" ;;
    esac
done

poetry run python experiments/simulate.py \
    --out "$output_dir" --reps "$reps" --reference-reps "$reference_reps" \
    --limit-reps "$limit_reps" --seed "$seed"
poetry run python experiments/verify.py --out "$output_dir"
exec poetry run python experiments/plot.py --input-dir "$output_dir" --figure-dir "$figure_dir"
