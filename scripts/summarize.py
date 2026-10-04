"""Aggregate evaluated metrics within datasets, then average datasets equally."""
import argparse
from pathlib import Path
import pandas as pd


def summarize(frame):
    metrics = ["VUS-PR", "Affiliation-F", "F1_T", "Standard-F1"]
    if any(key not in frame for key in metrics):
        raise ValueError("Input lacks evaluation metrics; run infer.py --evaluate first")
    frame = frame.copy()
    frame["dataset"] = frame.filename.str.split("_").str[1]
    if frame[metrics].isna().any().any():
        raise ValueError("Missing metric values must be resolved before aggregation")
    if frame.duplicated(["filename", "head"]).any():
        raise ValueError("Duplicate filename/head rows")
    per_dataset = frame.groupby(["head", "dataset"])[metrics].mean() * 100
    return per_dataset, per_dataset.groupby("head").mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    per_dataset, macro = summarize(pd.read_csv(args.metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(args.output_dir / "per_dataset_percent.csv")
    macro.to_csv(args.output_dir / "macro_percent.csv")
    print(macro.to_string())
