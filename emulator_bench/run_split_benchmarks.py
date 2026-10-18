import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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


def train(cmd, slots):
    gpu = slots.get()
    try:
        env = None if gpu is None else {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    finally:
        slots.put(gpu)


def main():
    parser = argparse.ArgumentParser(description="Train EITLEM-Kinetics (original settings) on every default EMULaToR split, for every seed.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device when --gpus is not set.")
    parser.add_argument("--gpus", nargs="+", default=None, help="Physical GPU ids, e.g. --gpus 0 1 2. Each run sees one GPU as cuda:0.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent runs per GPU (per --device when --gpus is not set).")
    parser.add_argument("--amp", action="store_true", help="Mixed precision: bf16 on Ampere or newer, fp16 on older GPUs. Default is fp32.")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=100, help="Original value is 100. Change it only for smoke tests.")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    run_device = "cuda:0" if args.gpus else args.device
    cache_env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus[0]} if args.gpus else None

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
                run_device,
            ],
            check=True,
            cwd=str(REPO_ROOT),
            env=cache_env,
        )

    runs = [(task, job, seed) for task in args.tasks for job in discover_split_jobs(data_root / task, args.split_groups) for seed in args.seeds]
    if not runs:
        raise FileNotFoundError(f"No split jobs found under {data_root}")

    commands = []
    for task, job, seed in runs:
        out_dir = output_dir / task / job["split_group"] / job["split_name"] / f"seed_{seed}"
        if (out_dir / "final_results_test.csv").exists() and not args.overwrite:
            continue
        cmd = [
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
            run_device,
            "--num_workers",
            str(args.num_workers),
            "--seed",
            str(seed),
        ]
        commands.append(cmd + ["--amp"] if args.amp else cmd)

    slots = queue.Queue()
    for gpu in args.gpus or [None]:
        for _ in range(args.runs_per_gpu):
            slots.put(None if gpu is None else str(gpu))
    with ThreadPoolExecutor(slots.qsize()) as pool:
        list(progress(pool.map(lambda cmd: train(cmd, slots), commands), desc="Benchmark runs", unit="run", total=len(commands)))


if __name__ == "__main__":
    main()
