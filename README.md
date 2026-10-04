# TS-Router

Code and pretrained checkpoints for **Generalist Representation, Specialist Detection: TS-Router for Time Series Anomaly Detection**.

TS-Router uses a frozen time-series encoder to select specialist anomaly detectors and fuse their scores. It supports univariate and multivariate time series.

## Installation

Download the repository from the [anonymous project page](https://anonymous.4open.science/r/TS-Router-D8FF/) and enter the project directory.

Use Python 3.10:

```bash
pip install -r requirements.txt
```

Pretrained checkpoints are included in `checkpoints/`. No encoder or router training is required.

## Quick Start

Run the univariate example:

```bash
python infer.py \
  --input examples/synthetic_univariate.csv \
  --train-index 512 \
  --device cpu \
  --evaluate \
  --output-dir results/example
```

To test both univariate and multivariate inputs with all routing heads:

```bash
bash scripts/smoke_test.sh cpu
```

Replace `cpu` with `cuda` for GPU execution.

## Custom Data

Provide a CSV with one numeric column per variable. An optional binary `Label` column is used only for evaluation. Exclude timestamps and other metadata columns.

```bash
python infer.py \
  --input /path/to/series.csv \
  --train-index 1000 \
  --output-dir results/custom
```

`--train-index` specifies the reference-prefix length; anomaly scores are returned for the remaining observations. Add `--evaluate` when labels are available.

The default configuration selects **3 of 11 specialists** using the **VUS-PR** routing head. Use `--top-k` to change the number of specialists or `--head` to select `VUS-PR`, `Affiliation-F1`, `F1T`, `Standard-F1`, or `all`.

Outputs include anomaly scores (`*.scores.csv`), routing details (`*.routing.json`), and per-file results (`metrics.csv`).

## Benchmarks

Download TSB-AD:

```bash
python scripts/download_data.py --track U
python scripts/download_data.py --track M
```

Point `--input` to the extracted CSV directory and use the corresponding file list:

```bash
python infer.py \
  --input /path/to/TSB-AD-U \
  --file-list data/univariate.csv \
  --evaluate \
  --output-dir results/univariate
```

For multivariate evaluation, use `data/multivariate.csv`.

Evaluation reports VUS-PR, Affiliation-F, F1-T, and standard F1. F1-based metrics use label-based threshold optimization. Inference settings are recorded in `configs/default.json`.

## Tests and Acknowledgments

```bash
python -m unittest discover -s tests
```

See `examples/expected_results.json` for reference outputs and `THIRD_PARTY_NOTICES.txt` for third-party attribution.
