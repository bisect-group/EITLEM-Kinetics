import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    EMULATOR_ROOT,
    TASKS,
    discover_split_jobs,
)

if str(EMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EMULATOR_ROOT))
from src.utils.rich_progress import progress

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def main():
    parser = argparse.ArgumentParser(description="Train EITLEM-Kinetics (original settings) on every default EMULaToR split, for every seed.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=100, help="Original value is 100. Change it only for smoke tests.")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"

    if not args.skip_cache:
        subprocess.run(
            [
                sys.executable,
                str(CACHE_SCRIPT),
                "--tasks",
                *args.tasks,
                "--data_root",
                str(data_root),
                "--embeddings_dir",
                str(embeddings_dir),
                "--split_groups",
                *args.split_groups,
                "--device",
                args.device,
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    runs = [(task, job, seed) for task in args.tasks for job in discover_split_jobs(data_root / task, args.split_groups) for seed in args.seeds]
    if not runs:
        raise FileNotFoundError(f"No split jobs found under {data_root}")

    for task, job, seed in progress(runs, desc="Benchmark runs", unit="run"):
        out_dir = output_dir / task / "results" / job["split_group"] / job["split_name"] / f"seed_{seed}"
        if (out_dir / "final_results_test.csv").exists() and not args.overwrite:
            continue
        subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--train_path",
                job["train_path"],
                "--val_path",
                job["val_path"],
                "--test_path",
                job["test_path"],
                "--embeddings_dir",
                str(embeddings_dir),
                "--out_dir",
                str(out_dir),
                "--task_name",
                f"{task}_{job['split_group']}_{job['split_name']}_seed{seed}",
                "--predictor_type",
                task,
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
                "--seed",
                str(seed),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )


if __name__ == "__main__":
    main()
