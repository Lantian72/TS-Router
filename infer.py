"""Run frozen checkpoints, selected specialists, and optional label-based evaluation."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from ts_router import ROOT, TSRouter


def read_input(path, train_index=None):
    frame = pd.read_csv(path).dropna()
    labels = frame["Label"].to_numpy(dtype=int) if "Label" in frame else None
    data = frame.drop(columns=["Label"], errors="ignore").to_numpy(dtype=float)
    if train_index is None:
        match = re.search(r"_tr_(\d+)(?:_|\.)", Path(path).name)
        if not match:
            raise ValueError("Supply --train-index or a filename containing _tr_<index>_")
        train_index = int(match.group(1))
    return data, labels, train_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Numeric CSV or directory; Label column is optional")
    parser.add_argument("--file-list", type=Path, help="CSV manifest with a filename column, used with an input directory")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--train-index", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--head", choices=["VUS-PR", "Affiliation-F1", "F1T", "Standard-F1", "all"], default="VUS-PR")
    parser.add_argument("--route-span", choices=["original", "full", "test"], default="original")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="Compute metrics after scoring; requires Label")
    args = parser.parse_args()
    if args.file_list:
        names = pd.read_csv(args.file_list)["filename"].tolist()
        if any(Path(str(name)).name != name for name in names):
            raise ValueError("File manifests must contain filenames without directories")
        paths = [args.input / name for name in names]
    else:
        paths = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Input file list is empty or contains missing files")
    model = TSRouter(args.checkpoint_dir, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        data, labels, split = read_input(path, args.train_index)
        if args.evaluate and labels is None:
            raise ValueError("--evaluate requires a Label column: " + str(path))
        result = model.predict(data, split, args.top_k, args.head, args.route_span, args.seed)
        metadata = {k: v for k, v in result.items() if k != "scores"}
        (args.output_dir / (path.stem + ".routing.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        for head, scores in result["scores"].items():
            output = pd.DataFrame({"index": np.arange(split, split + len(scores)), "anomaly_score": scores})
            if labels is not None:
                output["Label"] = labels[split:]
            output.to_csv(args.output_dir / (path.stem + "." + head + ".scores.csv"), index=False)
            row = {"filename": path.name, "head": head, "n_test": len(scores),
                   "selected_experts": json.dumps(result["selected_experts"][head])}
            if args.evaluate:
                from evaluation.metrics import get_metrics
                from utils.slidingWindows import find_length_rank
                metrics = get_metrics(scores, labels[split:], slidingWindow=find_length_rank(data, rank=1))
                row.update({k: float(metrics[k]) for k in ["VUS-PR", "Affiliation-F", "F1_T", "Standard-F1"]})
            rows.append(row)
            print(json.dumps(row))
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
