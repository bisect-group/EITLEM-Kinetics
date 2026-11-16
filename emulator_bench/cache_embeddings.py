import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    CACHE_VERSION,
    DEFAULT_SPLIT_GROUPS,
    EMULATOR_ROOT,
    ESM_MODEL,
    LONG_SEQ_STRIDE,
    MAX_SEQ_LEN,
    TASKS,
    discover_split_jobs,
    ensure_parent,
    ligand_cache_path,
    normalize_sequence,
    protein_cache_dir,
    protein_cache_key,
    read_table,
    save_json,
)
from emulator_bench.feature_pipeline import (
    build_esm_batches,
    embed_long_sequence,
    ligand_cache_item,
    ProteinEmbeddingStore,
    load_esm1v_model,
    _esm_forward,
)

if str(EMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EMULATOR_ROOT))
from src.utils.rich_progress import progress


SHARD_RESIDUES = 1_000_000  # about 5 GB per shard


def _collect_unique_values(jobs, sequence_col: str, smiles_col: str):
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[split_key]))
            if sequence_col not in frame.columns or smiles_col not in frame.columns:
                raise ValueError(f"Expected columns `{sequence_col}` and `{smiles_col}` in {job[split_key]}")
            sequences.update(normalize_sequence(value) for value in frame[sequence_col].astype(str))
            smiles_values.update(str(value) for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def _save_npz(path: Path, item: dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **item)
    tmp_path.replace(path)


def _save_protein_shard(directory: Path, embeddings: dict) -> None:
    """Write one shard atomically: the `.json` index appears last, so a shard without it is ignored and rewritten."""
    directory.mkdir(parents=True, exist_ok=True)
    name = f"shard_{len(list(directory.glob('shard_*.json'))):05d}"
    total = sum(len(embedding) for embedding in embeddings.values())
    tmp_npy = directory / f"{name}.npy.tmp"
    residues = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=np.float32, shape=(total, next(iter(embeddings.values())).shape[1]))
    index, offset = {}, 0
    for sequence, embedding in embeddings.items():
        residues[offset : offset + len(embedding)] = embedding
        index[protein_cache_key(sequence)] = [offset, len(embedding)]
        offset += len(embedding)
    residues.flush()
    del residues
    tmp_npy.replace(directory / f"{name}.npy")
    tmp_json = directory / f"{name}.json.tmp"
    tmp_json.write_text(json.dumps(index))
    tmp_json.replace(directory / f"{name}.json")


def cache_proteins(args, sequences):
    directory = protein_cache_dir(args.embeddings_dir)
    if args.overwrite:
        for stale in directory.glob("shard_*"):
            stale.unlink()
    cached = ProteinEmbeddingStore(args.embeddings_dir).index
    pending = [seq for seq in sequences if protein_cache_key(seq) not in cached]
    if not pending:
        print("Protein cache is already complete.")
        return {"proteins_total": len(sequences), "proteins_written": 0}

    device = torch.device(args.device)
    print(f"Protein cache device: {device} | precision: fp32")
    model, batch_converter = load_esm1v_model(device)
    batches = build_esm_batches(
        pending,
        max_residues=args.max_residues,
        max_seq_len=MAX_SEQ_LEN,
        max_batch=args.max_batch,
    )

    written = 0
    buffered = {}
    iterator = progress(batches, desc="Caching protein embeddings", unit="batch")
    for batch in iterator:
        if len(batch) == 1 and len(batch[0]) > MAX_SEQ_LEN:
            sequence = batch[0]
            buffered[sequence] = embed_long_sequence(
                model,
                batch_converter,
                sequence,
                device=device,
                max_window=MAX_SEQ_LEN,
                stride=LONG_SEQ_STRIDE,
            )
        else:
            buffered.update(_esm_forward(model, batch_converter, batch, device=device))
        written += len(batch)
        if sum(len(embedding) for embedding in buffered.values()) >= SHARD_RESIDUES or written == len(pending):
            _save_protein_shard(directory, buffered)
            buffered = {}
        iterator.set_postfix(written=written, remaining=len(pending) - written)

    return {"proteins_total": len(sequences), "proteins_written": written}


def cache_ligands(args, smiles_values):
    pending = [
        smiles
        for smiles in smiles_values
        if args.overwrite
        or not ligand_cache_path(
            args.embeddings_dir,
            smiles,
            mol_type=args.mol_type,
            radius=args.radius,
            nbits=args.nbits,
        ).exists()
    ]
    if not pending:
        print("Ligand cache is already complete.")
        return {"ligands_total": len(smiles_values), "ligands_written": 0}

    written = 0
    iterator = progress(pending, desc="Caching molecular fingerprints", unit="smiles")
    for smiles in iterator:
        item = ligand_cache_item(smiles, mol_type=args.mol_type, radius=args.radius, nbits=args.nbits)
        _save_npz(
            ligand_cache_path(
                args.embeddings_dir,
                smiles,
                mol_type=args.mol_type,
                radius=args.radius,
                nbits=args.nbits,
            ),
            item,
        )
        written += 1
        iterator.set_postfix(written=written, remaining=len(pending) - written)
    return {"ligands_total": len(smiles_values), "ligands_written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache reusable EITLEM-Kinetics protein embeddings and molecular fingerprints.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT / "embeddings"))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mol_type", choices=["MACCSKeys", "ECFP", "RDKIT"], default="MACCSKeys")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--max_residues", type=int, default=12000)
    parser.add_argument("--max_batch", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.data_root = Path(args.data_root)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    jobs = [job for task in args.tasks for job in discover_split_jobs(args.data_root / task, args.split_groups)]
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.data_root}")

    started = time.time()
    sequences, smiles_values = _collect_unique_values(jobs, sequence_col=args.sequence_col, smiles_col=args.smiles_col)
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique normalized sequences: {len(sequences)}")
    print(f"Unique smiles: {len(smiles_values)}")

    protein_stats = cache_proteins(args, sequences)
    ligand_stats = cache_ligands(args, smiles_values)

    manifest = {
        "cache_version": CACHE_VERSION,
        "data_root": str(args.data_root),
        "tasks": list(args.tasks),
        "embeddings_dir": str(args.embeddings_dir),
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "split_groups": list(args.split_groups),
        "mol_type": args.mol_type,
        "radius": int(args.radius),
        "nbits": int(args.nbits),
        "protein_dtype": "float32",
        "protein_model": ESM_MODEL,
        "protein_max_seq_len": MAX_SEQ_LEN,
        "protein_long_seq_stride": LONG_SEQ_STRIDE,
        "protein_cache": protein_stats,
        "ligand_cache": ligand_stats,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.embeddings_dir / "manifest.json", manifest)
    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
