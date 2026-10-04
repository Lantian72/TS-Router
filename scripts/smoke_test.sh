#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python infer.py --input examples/synthetic_univariate.csv --train-index 512 --head all --device "${1:-cpu}" --evaluate --output-dir results/smoke_univariate
python infer.py --input examples/synthetic_multivariate.csv --train-index 512 --head all --device "${1:-cpu}" --evaluate --output-dir results/smoke_multivariate
