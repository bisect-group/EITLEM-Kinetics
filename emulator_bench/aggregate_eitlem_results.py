"""Aggregate EITLEM-Kinetics results over seeds.

Reads `<output_dir>/<task>/results/<split_group>/<split_name>/seed_<n>/final_results_{val,test}.csv`
and writes `<output_dir>/aggregate_eitlem_results.csv` with the mean and std of every metric.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_OUTPUT_ROOT

METRICS = ["mae", "mse", "rmse", "r2_score", "pearson", "spearman", "loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    rows = []
    for test_csv in sorted(output_dir.glob("*/results/*/*/seed_*/final_results_test.csv")):
        seed_dir = test_csv.parent
        task, _, split_group, split_name, seed = seed_dir.relative_to(output_dir).parts
        for partition in ("val", "test"):
            metrics = pd.read_csv(seed_dir / f"final_results_{partition}.csv").iloc[0]
            rows.append(
                {"task": task, "split_group": split_group, "split_name": split_name, "partition": partition, "seed": seed, **metrics[METRICS]}
            )
    if not rows:
        raise FileNotFoundError(f"No finished runs under {output_dir}")

    runs = pd.DataFrame(rows)
    grouped = runs.groupby(["task", "split_group", "split_name", "partition"], sort=False)[METRICS].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped["n_seeds"] = runs.groupby(["task", "split_group", "split_name", "partition"], sort=False).size()
    out_path = output_dir / "aggregate_eitlem_results.csv"
    grouped.reset_index().to_csv(out_path, index=False)
    print(f"Saved {len(grouped)} rows to {out_path}")


if __name__ == "__main__":
    main()
