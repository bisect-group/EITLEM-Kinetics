import sys
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import esm
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


RDLogger.DisableLog("rdApp.*")
RDKIT_FPGEN = AllChem.GetRDKitFPGenerator(fpSize=1024)


def load_esm1v_model(device: torch.device):
    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model.eval()
    model = model.to(device)
    return model, alphabet.get_batch_converter()


def build_esm_batches(
    sequences: Sequence[str],
    max_residues: int = 12000,
    max_seq_len: int = 1022,
    max_batch: int = 16,
) -> List[List[str]]:
    ordered = sorted([normalize_sequence(seq) for seq in sequences], key=len, reverse=True)
    batches: List[List[str]] = []
    batch: List[str] = []
    batch_residues = 0

    for sequence in ordered:
        seq_len = len(sequence)
        if seq_len > max_seq_len:
            if batch:
                batches.append(batch)
                batch = []
                batch_residues = 0
            batches.append([sequence])
            continue

        if batch and (len(batch) >= max_batch or batch_residues + seq_len > max_residues):
            batches.append(batch)
            batch = []
            batch_residues = 0

        batch.append(sequence)
        batch_residues += seq_len

    if batch:
        batches.append(batch)
    return batches


def _esm_forward(model, batch_converter, sequences: Sequence[str], device: torch.device) -> Dict[str, np.ndarray]:
    labels = [(f"seq_{idx}", seq) for idx, seq in enumerate(sequences)]
    _, batch_strs, tokens = batch_converter(labels)
    tokens = tokens.to(device, non_blocking=True)
    with torch.no_grad():
        result = model(tokens, repr_layers=[33], return_contacts=False)
    representations = result["representations"][33].detach().cpu().float()
    embedded: Dict[str, np.ndarray] = {}
    for idx, sequence in enumerate(batch_strs):
        embedded[str(sequence)] = representations[idx, 1 : len(sequence) + 1].numpy()
    return embedded


def embed_long_sequence(
    model,
    batch_converter,
    sequence: str,
    device: torch.device,
    max_window: int = 1022,
    stride: int = 800,
) -> np.ndarray:
    if len(sequence) <= max_window:
        return _esm_forward(model, batch_converter, [sequence], device=device)[sequence]

    if stride >= max_window:
        raise ValueError("stride must be smaller than max_window for long-sequence ESM extraction")

    accum = None
    counts = np.zeros((len(sequence), 1), dtype=np.float32)
    start = 0
    while start < len(sequence):
        end = min(start + max_window, len(sequence))
        window_sequence = sequence[start:end]
        window_embedding = _esm_forward(
            model,
            batch_converter,
            [window_sequence],
            device=device,
        )[window_sequence].astype(np.float32, copy=False)
        if accum is None:
            accum = np.zeros((len(sequence), window_embedding.shape[-1]), dtype=np.float32)
        accum[start:end] += window_embedding
        counts[start:end] += 1.0
        if end >= len(sequence):
            break
        start += stride
    accum /= counts
    return accum


def featurize_smiles(smiles: str, mol_type: str = "MACCSKeys", radius: int = 4, nbits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    mol_type = str(mol_type)
    if mol_type == "ECFP":
        return np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits).ToList(), dtype=np.uint8)
    if mol_type == "MACCSKeys":
        return np.asarray(MACCSkeys.GenMACCSKeys(mol).ToList(), dtype=np.uint8)
    if mol_type == "RDKIT":
        generator = RDKIT_FPGEN if int(nbits) == 1024 else AllChem.GetRDKitFPGenerator(fpSize=int(nbits))
        return np.asarray(generator.GetFingerprint(mol).ToList(), dtype=np.uint8)
    raise ValueError(f"Unsupported mol_type: {mol_type}")


def protein_cache_item(embedding: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "embedding": embedding.astype(np.float32, copy=False),
        "length": np.asarray([embedding.shape[0]], dtype=np.int32),
    }


def ligand_cache_item(smiles: str, mol_type: str = "MACCSKeys", radius: int = 4, nbits: int = 1024) -> Dict[str, np.ndarray]:
    fingerprint = featurize_smiles(smiles, mol_type=mol_type, radius=radius, nbits=nbits)
    return {
        "fingerprint": fingerprint.astype(np.uint8, copy=False),
        "feature_dim": np.asarray([fingerprint.shape[0]], dtype=np.int32),
    }


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(
            f"Corrupted cache file: {path}. Rebuild with `cache_embeddings.py --overwrite`."
        ) from exc


class ProteinEmbeddingStore:
    def __init__(self, embeddings_dir: Path, max_items: int = 512):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def get(self, sequence: str) -> Dict[str, np.ndarray]:
        normalized = normalize_sequence(sequence)
        if normalized in self._cache:
            self._cache.move_to_end(normalized)
            return self._cache[normalized]

        path = protein_cache_path(self.embeddings_dir, normalized)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached protein embedding: {path}")
        item = load_npz(path)
        self._cache[normalized] = item
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


class LigandEmbeddingStore:
    def __init__(
        self,
        embeddings_dir: Path,
        mol_type: str,
        radius: int,
        nbits: int,
        smiles_values: Optional[Sequence[str]] = None,
        preload: bool = True,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.mol_type = mol_type
        self.radius = int(radius)
        self.nbits = int(nbits)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        if preload and smiles_values is not None:
            for smiles in sorted({str(value) for value in smiles_values}):
                path = ligand_cache_path(
                    self.embeddings_dir,
                    smiles,
                    mol_type=self.mol_type,
                    radius=self.radius,
                    nbits=self.nbits,
                )
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached ligand embedding: {path}")
                self._cache[smiles] = load_npz(path)

    def get(self, smiles: str) -> Dict[str, np.ndarray]:
        smiles = str(smiles)
        if smiles in self._cache:
            return self._cache[smiles]
        path = ligand_cache_path(
            self.embeddings_dir,
            smiles,
            mol_type=self.mol_type,
            radius=self.radius,
            nbits=self.nbits,
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ligand embedding: {path}")
        item = load_npz(path)
        self._cache[smiles] = item
        return item
