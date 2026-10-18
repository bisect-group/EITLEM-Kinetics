import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor


def mol_feature_dim(mol_type: str, nbits: int) -> int:
    mol_type = str(mol_type)
    if mol_type == "MACCSKeys":
        return 167
    if mol_type in {"ECFP", "RDKIT"}:
        return int(nbits)
    raise ValueError(f"Unsupported mol_type: {mol_type}")


def build_model(
    predictor_type: str,
    mol_type: str,
    nbits: int,
    hidden_dim: int,
    protein_dim: int,
    layer: int,
    dropout: float,
    att_layer: int,
):
    in_dim = mol_feature_dim(mol_type=mol_type, nbits=nbits)
    predictor_type = str(predictor_type).lower()
    if predictor_type == "kcat":
        return EitlemKcatPredictor(in_dim, hidden_dim, protein_dim, layer, dropout, att_layer)
    if predictor_type == "km":
        return EitlemKmPredictor(in_dim, hidden_dim, protein_dim, layer, dropout, att_layer)
    raise ValueError(f"Unsupported predictor_type: {predictor_type}")
