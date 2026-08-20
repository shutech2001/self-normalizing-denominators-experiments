#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_DIR"

exec poetry run python experiments/real_data_experiments.py "$@"
