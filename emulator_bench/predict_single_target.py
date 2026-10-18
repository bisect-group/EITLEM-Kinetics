import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_OUTPUT_ROOT, read_table, regression_metrics, require_columns, save_json
from emulator_bench.dataset import EitlemDataLoader
from emulator_bench.feature_pipeline import LigandEmbeddingStore, ProteinEmbeddingStore
from emulator_bench.modeling import build_model


def main():
    parser = argparse.ArgumentParser(description="Run inference for one split file using a trained emulator bench checkpoint.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT / "embeddings"))
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4, help="Threads that build batches.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Batches built ahead.")
    parser.add_argument("--lazy_ligands", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    train_args = checkpoint["args"]
    device = torch.device(args.device)

    frame = read_table(Path(args.input_path))
    require_columns(frame, [args.sequence_col, args.smiles_col], Path(args.input_path))
    protein_store = ProteinEmbeddingStore(Path(args.embeddings_dir))
    ligand_store = LigandEmbeddingStore(
        Path(args.embeddings_dir),
        mol_type=train_args["mol_type"],
        radius=int(train_args["radius"]),
        nbits=int(train_args["nbits"]),
        smiles_values=frame[args.smiles_col].astype(str).tolist(),
        preload=not args.lazy_ligands,
    )
    loader = EitlemDataLoader(
        frame,
        protein_store,
        ligand_store,
        device,
        args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
    )

    model = build_model(
        predictor_type=train_args["predictor_type"],
        mol_type=train_args["mol_type"],
        nbits=int(train_args["nbits"]),
        hidden_dim=512,
        protein_dim=1280,
        layer=10,
        dropout=0.5,
        att_layer=10,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds = []
    truths = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch)
            preds.append(prediction.detach().cpu().float())
            if batch.value is not None:
                truths.append(batch.value.detach().cpu().float())

    pred_np = torch.cat(preds).numpy() if preds else np.array([], dtype=np.float32)
    output = frame.copy()
    output["prediction"] = pred_np
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_csv, index=False)

    metrics = {}
    if truths:
        truth_np = torch.cat(truths).numpy()
        metrics = regression_metrics(truth_np, pred_np)
        metrics_path = out_csv.with_name(out_csv.stem + "_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        save_json(
            out_csv.with_name(out_csv.stem + "_manifest.json"),
            {
                "input_path": args.input_path,
                "checkpoint_path": args.ckpt_path,
                "output_csv": str(out_csv),
                "metrics_csv": str(metrics_path),
                "precision_mode": "fp32",
            },
        )


if __name__ == "__main__":
    main()
