import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    EMULATOR_ROOT,
    append_csv_row,
    read_table,
    regression_metrics,
    require_columns,
    save_json,
    set_seed,
)
from emulator_bench.dataset import CachedEitlemDataset, EitlemDataLoader
from emulator_bench.feature_pipeline import LigandEmbeddingStore, ProteinEmbeddingStore
from emulator_bench.modeling import build_model

if str(EMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EMULATOR_ROOT))
from src.utils.rich_progress import progress


def _prepare_batch(batch, device: torch.device):
    batch = batch.to(device)
    batch.x = batch.x.float()
    batch.pro_emb = batch.pro_emb.float()
    if hasattr(batch, "value"):
        batch.value = batch.value.float()
    return batch


def evaluate_loader(model, loader, device, desc="Evaluation", show_progress=True):
    model.eval()
    preds, truths = [], []
    total_loss = 0.0
    total_samples = 0
    loss_fn = torch.nn.MSELoss()
    iterator = progress(loader, desc=desc, unit="batch", leave=False) if show_progress else loader
    with torch.no_grad():
        for batch in iterator:
            batch = _prepare_batch(batch, device)
            prediction = model(batch)
            total_loss += float(loss_fn(prediction, batch.value).item()) * int(batch.num_graphs)
            total_samples += int(batch.num_graphs)
            preds.append(prediction.detach().cpu().float())
            truths.append(batch.value.detach().cpu().float())
    truth_np = torch.cat(truths).numpy()
    pred_np = torch.cat(preds).numpy()
    metrics = regression_metrics(truth_np, pred_np)
    metrics["loss"] = total_loss / max(1, total_samples)
    return truth_np, pred_np, metrics


def train_one_epoch(model, loader, optimizer, device, desc="Train"):
    model.train()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    for batch in progress(loader, desc=desc, unit="batch", leave=False):
        batch = _prepare_batch(batch, device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch.value)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch.num_graphs)
        total_samples += int(batch.num_graphs)
    return total_loss / max(1, total_samples)


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train EITLEM-Kinetics with its original settings on explicit train/val/test files.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="eitlem")
    parser.add_argument("--predictor_type", choices=["kcat", "km"], required=True)
    parser.add_argument("--mol_type", choices=["MACCSKeys", "ECFP", "RDKIT"], default="MACCSKeys")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    # Original settings: Code/iter_train_scripts.py (kineticsTrainer, no pre-trained weights).
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=666)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--prefetch_factor", type=int, default=5)
    parser.add_argument("--protein_cache_items", type=int, default=512)
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: Path(getattr(args, f"{name}_path")) for name in ("train", "val", "test")}
    frames = {name: read_table(path) for name, path in paths.items()}
    for name, frame in frames.items():
        require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], paths[name])
    all_frame = pd.concat(frames.values(), ignore_index=True)

    protein_store = ProteinEmbeddingStore(Path(args.embeddings_dir), max_items=args.protein_cache_items)
    ligand_store = LigandEmbeddingStore(
        Path(args.embeddings_dir),
        mol_type=args.mol_type,
        radius=args.radius,
        nbits=args.nbits,
        smiles_values=all_frame[args.smiles_col].astype(str).tolist(),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    loaders = {}
    for name, frame in frames.items():
        dataset = CachedEitlemDataset(frame, protein_store, ligand_store, args.sequence_col, args.smiles_col, args.target_col)
        loaders[name] = EitlemDataLoader(dataset, shuffle=(name == "train"), **loader_kwargs)

    model = build_model(
        predictor_type=args.predictor_type,
        mol_type=args.mol_type,
        nbits=args.nbits,
        hidden_dim=512,
        protein_dim=1280,
        layer=10,
        dropout=0.5,
        att_layer=10,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = time.time()
    print(f"Device: {device} | precision: fp32", flush=True)

    for epoch in progress(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row["train_loss"] = train_one_epoch(model, loaders["train"], optimizer, device, desc=f"Epoch {epoch} train")
        _, _, val_metrics = evaluate_loader(model, loaders["val"], device, show_progress=False)
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["elapsed_seconds"] = time.time() - started
        scheduler.step()
        append_csv_row(out_dir / "logfile.csv", row)

    # The original code keeps only the weights of the last epoch.
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, out_dir / "final_model.pth")

    final_metrics = {}
    for name in ("train", "val", "test"):
        y_true, y_pred, metrics = evaluate_loader(model, loaders[name], device, desc=f"Final {name}")
        save_predictions(out_dir / f"pred_label_{name}.csv", y_true, y_pred)
        pd.DataFrame([metrics]).to_csv(out_dir / f"final_results_{name}.csv", index=False)
        final_metrics[name] = metrics

    save_json(
        out_dir / "run_summary.json",
        {
            "task_name": args.task_name,
            "started_at": started_at,
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - started,
            "precision_mode": "fp32",
            "train_path": str(paths["train"]),
            "val_path": str(paths["val"]),
            "test_path": str(paths["test"]),
            "final_train_metrics": final_metrics["train"],
            "final_val_metrics": final_metrics["val"],
            "final_test_metrics": final_metrics["test"],
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
