"""The wrapper reproduces the authors' training path (Code/iter_train_scripts.py, kineticsTrainer).

1. EitlemDataLoader yields the same batches as the original PyG DataLoader (shuffle, persistent workers).
2. The wrapper's `fit` gives the same losses and weights as the original Trainer/Tester loop.
3. The cached ESM1v features equal the original extraction (Code/extract.py).

Everything runs on CPU in fp32. Test 2 uses the original model classes with smaller dimensions (hidden 64, 3 layers) for speed. The CUDA-only changes (TF32, cudnn.benchmark, side-stream copies) are not covered.
Test 3 needs the ESM1v weights in `$TORCH_HOME/hub/checkpoints` and skips without them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench import train_single_target_tvt as trainer  # noqa: E402
from emulator_bench.cache_embeddings import _save_protein_shard  # noqa: E402
from emulator_bench.common import protein_cache_dir  # noqa: E402
from emulator_bench.dataset import EitlemDataLoader  # noqa: E402
from emulator_bench.feature_pipeline import (  # noqa: E402
    ProteinEmbeddingStore,
    _esm_forward,
    build_esm_batches,
    ligand_cache_item,
    load_esm1v_model,
)
from emulator_bench.modeling import build_model  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "Code"))
import dataset as original_dataset  # noqa: E402
import eitlem_utils as original_utils  # noqa: E402

assert Path(original_dataset.__file__).parent == REPO_ROOT / "Code", "`dataset` is not the authors' module"

SMILES = ["CCO", "CC(=O)O", "c1ccccc1O", "OC(=O)CC(O)(CC(=O)O)C(=O)O", "NCC(=O)O", "CC(C)CO", "C1CCCCC1", "OCC1OC(O)C(O)C(O)C1O"]
BATCH_SIZE = 8
EPOCHS = 2
PROTEIN_DIM = 1280


def make_rows(n_rows, seed=0):
    """Rows of (sequence, smiles, log10 value); sequences repeat and vary in length."""
    rng = np.random.default_rng(seed)
    sequences = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=4 + 3 * i)) for i in range(9)]
    return pd.DataFrame(
        {
            "sequence": [sequences[i] for i in rng.integers(0, len(sequences), n_rows)],
            "smiles": [SMILES[i] for i in rng.integers(0, len(SMILES), n_rows)],
            "log10_value": rng.integers(-256, 256, n_rows) / 64,  # exact in fp32, so log10(10**v) rounds back to v
        }
    )


class FakeProteins:
    """Per-sequence residue embeddings; the wrapper store and the original `.pt` files hold the same arrays."""

    def __init__(self, dim=PROTEIN_DIM):
        self.dim = dim

    def get(self, sequence):
        return np.random.default_rng(sum(map(ord, sequence))).standard_normal((len(sequence), self.dim)).astype(np.float32)


class FakeLigands:
    def get(self, smiles):
        return ligand_cache_item(smiles)


def original_loader(frame, tmp_path, shuffle, proteins):
    """The original DataSet and DataLoader with their `iter_train_scripts.py` options (30 workers reduced to 2)."""
    pro_dir = tmp_path / "proteins"
    pro_dir.mkdir(exist_ok=True)
    pair_info = []
    for sequence, smiles, value in zip(frame["sequence"], frame["smiles"], frame["log10_value"]):
        torch.save(torch.from_numpy(proteins.get(sequence)), pro_dir / f"{sequence}.pt")
        pair_info.append([sequence, smiles, 10.0**value])
    data = original_dataset.EitlemDataSet(pair_info, str(pro_dir), {smiles: smiles for smiles in SMILES}, 1024, 4, True, "MACCSKeys")
    return original_dataset.EitlemDataLoader(
        data=data, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False, num_workers=2, prefetch_factor=5, persistent_workers=True
    )


def wrapper_loader(frame, shuffle, proteins):
    return EitlemDataLoader(frame, proteins, FakeLigands(), torch.device("cpu"), BATCH_SIZE, shuffle=shuffle, num_workers=2)


def test_batches_match_original_dataloader(tmp_path):
    frame, proteins = make_rows(45), FakeProteins(dim=6)  # 6 batches, the last one partial
    torch.manual_seed(7)
    loader = original_loader(frame, tmp_path, True, proteins)
    original = [list(loader) for _ in range(3)]
    torch.manual_seed(7)
    loader = wrapper_loader(frame, True, proteins)
    wrapped = [list(loader) for _ in range(3)]
    assert len(loader) == len(original[0])
    for original_epoch, wrapped_epoch in zip(original, wrapped):
        for reference, batch in zip(original_epoch, wrapped_epoch):
            assert "edge_index" not in reference.keys()
            assert torch.equal(reference.batch, torch.arange(batch.num_graphs))  # one node per graph
            assert reference.num_graphs == batch.num_graphs
            assert torch.equal(reference.x, batch.x)
            assert torch.equal(reference.pro_emb, batch.pro_emb)
            assert torch.equal(reference.pro_emb_batch, batch.pro_emb_batch)
            assert torch.equal(reference.value, batch.value)
        assert len(original_epoch) == len(wrapped_epoch)
    assert not torch.equal(original[0][0].value, original[1][0].value)  # the order changes between epochs


@pytest.mark.parametrize("predictor_type", ["kcat", "km"])
def test_training_matches_original_loop(tmp_path, predictor_type):
    proteins = FakeProteins()
    train, val = make_rows(20, seed=1), make_rows(11, seed=2)
    device = torch.device("cpu")

    def new_model():
        torch.manual_seed(3)
        return build_model(predictor_type, "MACCSKeys", 1024, 64, PROTEIN_DIM, 3, 0.5, 3)

    def new_optimizer(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.9)

    # Original loop: Trainer.run, Tester.test, scheduler.step per epoch.
    model = new_model()
    optimizer, scheduler = new_optimizer(model)
    losses = []

    class RecordingMSE(torch.nn.MSELoss):
        def forward(self, prediction, target):
            losses.append(super().forward(prediction, target))
            return losses[-1]

    loss_fn = RecordingMSE()
    train_loader = original_loader(train, tmp_path, True, proteins)
    val_loader = original_loader(val, tmp_path, False, proteins)
    torch.manual_seed(5)
    original_trainer, original_tester = original_utils.Trainer(device, loss_fn, log10=True), original_utils.Tester(device, loss_fn, log10=True)
    for _ in range(EPOCHS):
        original_trainer.run(model, train_loader, optimizer, len(train), "train")
        original_tester.test(model, val_loader, len(val), "val")
        scheduler.step()
    original_weights = {key: value.clone() for key, value in model.state_dict().items()}
    n_train_batches = len(train_loader)
    train_losses = [[float(loss) for loss in losses[epoch * (n_train_batches + len(val_loader)) :][:n_train_batches]] for epoch in range(EPOCHS)]

    # Wrapper loop: fit (train_one_epoch, evaluate_loader, scheduler.step, checkpoint).
    wrapped = new_model()
    optimizer, scheduler = new_optimizer(wrapped)
    loaders = {"train": wrapper_loader(train, True, proteins), "val": wrapper_loader(val, False, proteins)}
    args = argparse.Namespace(
        task_name="t", train_path="a", val_path="b", test_path="c", seed=5, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3,
        amp=False, predictor_type=predictor_type, mol_type="MACCSKeys", radius=4, nbits=1024,
    )  # fmt: skip
    torch.manual_seed(5)
    trainer.fit(args, wrapped, optimizer, scheduler, None, loaders, device, None, tmp_path / "out")

    sizes = [BATCH_SIZE] * (len(train) // BATCH_SIZE) + [len(train) % BATCH_SIZE]
    log = pd.read_csv(tmp_path / "out" / "logfile.csv")
    for epoch, batch_losses in enumerate(train_losses):
        expected = sum(loss * size for loss, size in zip(batch_losses, sizes)) / len(train)
        assert log["train_loss"][epoch] == pytest.approx(expected, rel=1e-5)
    for key, value in wrapped.state_dict().items():
        torch.testing.assert_close(value, original_weights[key], rtol=1e-5, atol=1e-6, msg=key)
    assert log["lr"].tolist() == pytest.approx([1e-3, 9e-4])


def esm_checkpoint():
    path = Path(torch.hub.get_dir()) / "checkpoints" / "esm1v_t33_650M_UR90S_1.pt"
    if not path.exists():
        pytest.skip(f"ESM1v weights not found at {path}")


def test_cached_esm_features_match_original_extraction(tmp_path):
    esm_checkpoint()
    import esm

    sequences = ["MKTAYIAKQR", "GSHMLEDPVDAFKHLQ", "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHL"]
    device = torch.device("cpu")
    model, batch_converter = load_esm1v_model(device)

    # Original: Code/extract.py batches a FASTA file by token count and keeps residues 1..len of layer 33.
    fasta = tmp_path / "seq.fasta"
    fasta.write_text("".join(f">{i}\n{sequence}\n" for i, sequence in enumerate(sequences)))
    dataset = esm.FastaBatchedDataset.from_file(str(fasta))
    reference = {}
    loader = torch.utils.data.DataLoader(dataset, collate_fn=batch_converter, batch_sampler=dataset.get_batch_indices(4096, extra_toks_per_seq=1))
    with torch.no_grad():
        for labels, strs, tokens in loader:
            representations = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            for i, sequence in enumerate(strs):
                reference[sequence] = representations[i, 1 : len(sequence) + 1].clone().numpy()

    # Wrapper: the cache builder's batches, flat shard file, memory-mapped store.
    embeddings = {}
    for batch in build_esm_batches(sequences, max_residues=12000, max_seq_len=1022, max_batch=64):
        embeddings.update(_esm_forward(model, batch_converter, batch, device=device))
    _save_protein_shard(protein_cache_dir(tmp_path), embeddings)
    store = ProteinEmbeddingStore(tmp_path)
    for sequence in sequences:
        cached = store.get(sequence)
        assert cached.dtype == np.float32 and cached.shape == reference[sequence].shape
        np.testing.assert_allclose(cached, reference[sequence], rtol=1e-4, atol=1e-4)
