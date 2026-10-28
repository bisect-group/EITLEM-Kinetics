"""The wrapper reproduces the authors' training path (Code/iter_train_scripts.py, kineticsTrainer).

1. EitlemDataLoader yields the same batches as the original PyG DataLoader (shuffle, persistent workers).
2. The wrapper's `fit` gives the same losses and weights as the original Trainer/Tester loop.
3. The cached ESM1v features equal the original extraction (Code/extract.py).

Tests 1-3 run on CPU in fp32. Test 2 uses the original model classes with smaller dimensions (hidden 64, 3 layers) for speed.
The `gpu` tests cover the CUDA-only changes: side-stream copies in the loader, and the production flags (TF32, cudnn.benchmark) against deterministic fp32.
Test 3 runs on a tiny random ESM1v-shaped model. Its `esm1v` variant (marker `esm_weights`) needs the ESM1v weights in `$TORCH_HOME/hub/checkpoints` and skips without them; the verify script leaves it out.
"""

import argparse
import os
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
    embed_long_sequence,
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


def wrapper_loader(frame, shuffle, proteins, device="cpu"):
    return EitlemDataLoader(frame, proteins, FakeLigands(), torch.device(device), BATCH_SIZE, shuffle=shuffle, num_workers=2)


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


def tiny_esm1v():
    """Random ESM1v-shaped network: the same classes and 33 layers, 16-d (esm1v_t33_650M_UR90S_1 is 1280-d)."""
    import esm

    torch.manual_seed(0)
    alphabet = esm.data.Alphabet.from_architecture("roberta_large")
    args = argparse.Namespace(
        arch="roberta_large", layers=33, embed_dim=16, ffn_embed_dim=32, attention_heads=2, max_positions=1024, final_bias=True,
        emb_layer_norm_before=False, token_dropout=False,
    )  # fmt: skip
    return esm.model.esm1.ProteinBertModel(args, alphabet).eval(), alphabet.get_batch_converter()


def real_esm1v():
    path = Path(torch.hub.get_dir()) / "checkpoints" / "esm1v_t33_650M_UR90S_1.pt"
    if not path.exists():
        pytest.skip(f"ESM1v weights not found at {path}")
    return load_esm1v_model(torch.device("cpu"))


@pytest.mark.parametrize("load_model", [tiny_esm1v, pytest.param(real_esm1v, marks=pytest.mark.esm_weights)], ids=["tiny", "esm1v"])
def test_cached_esm_features_match_original_extraction(tmp_path, load_model):
    import esm

    sequences = ["MKTAYIAKQR", "GSHMLEDPVDAFKHLQ", "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHL"]
    device = torch.device("cpu")
    model, batch_converter = load_model()

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


def _original_features(model, batch_converter, tmp_path, sequences):
    """Code/extract.py path: FASTA, token-count batches, residues 1..len of layer 33."""
    import esm

    fasta = tmp_path / "ref.fasta"
    fasta.write_text("".join(f">{i}\n{sequence}\n" for i, sequence in enumerate(sequences)))
    dataset = esm.FastaBatchedDataset.from_file(str(fasta))
    loader = torch.utils.data.DataLoader(dataset, collate_fn=batch_converter, batch_sampler=dataset.get_batch_indices(4096, extra_toks_per_seq=1))
    reference = {}
    with torch.no_grad():
        for labels, strs, tokens in loader:
            representations = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            for i, sequence in enumerate(strs):
                reference[sequence] = representations[i, 1 : len(sequence) + 1].clone().numpy()
    return reference


def test_split_length_sorted_batches_match_original_extraction(tmp_path):
    """Production batching (max_residues 12000, max_batch 64) splits 40 sequences into several length-sorted batches."""
    rng = np.random.default_rng(0)
    sequences = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=int(n))) for n in rng.integers(300, 900, size=40)]
    model, batch_converter = tiny_esm1v()
    reference = _original_features(model, batch_converter, tmp_path, sequences)
    batches = build_esm_batches(sequences, max_residues=12000, max_seq_len=1022, max_batch=64)
    assert len(batches) > 1 and sum(map(len, batches)) == len(sequences)
    assert all(len(a[0]) >= len(b[0]) for a, b in zip(batches, batches[1:]))
    embeddings = {}
    for batch in batches:
        embeddings.update(_esm_forward(model, batch_converter, batch, device=torch.device("cpu")))
    assert set(embeddings) == set(reference)
    for sequence, expected in reference.items():
        np.testing.assert_allclose(embeddings[sequence], expected, rtol=1e-4, atol=1e-4)


def test_long_sequence_windows_average_original_window_features(tmp_path):
    """A 1900-residue protein is embedded in 1022-residue windows at stride 800; overlaps are averaged."""
    rng = np.random.default_rng(1)
    sequence = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=1900))
    model, batch_converter = tiny_esm1v()
    starts = [0, 800, 1600]
    windows = [sequence[s : s + 1022] for s in starts]
    reference = _original_features(model, batch_converter, tmp_path, windows)
    expected = np.zeros((len(sequence), 16), dtype=np.float32)
    counts = np.zeros((len(sequence), 1), dtype=np.float32)
    for start, window in zip(starts, windows):
        expected[start : start + len(window)] += reference[window]
        counts[start : start + len(window)] += 1
    batches = build_esm_batches([sequence], max_residues=12000, max_seq_len=1022, max_batch=64)
    assert batches == [[sequence]]
    cached = embed_long_sequence(model, batch_converter, sequence, device=torch.device("cpu"), max_window=1022, stride=800)
    np.testing.assert_allclose(cached, expected / counts, rtol=1e-4, atol=1e-4)


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.gpu
@needs_cuda
def test_side_stream_loader_yields_the_cpu_batches():
    """The pinned side-stream copies give the same batches, in the same order, as the CPU loader."""
    frame, proteins = make_rows(45), FakeProteins(dim=6)
    torch.manual_seed(7)
    loader = wrapper_loader(frame, True, proteins)
    cpu = [list(loader) for _ in range(2)]
    torch.manual_seed(7)
    loader = wrapper_loader(frame, True, proteins, "cuda")
    gpu = [list(loader) for _ in range(2)]
    for cpu_epoch, gpu_epoch in zip(cpu, gpu):
        assert len(cpu_epoch) == len(gpu_epoch)
        for reference, batch in zip(cpu_epoch, gpu_epoch):
            for name in ("x", "pro_emb", "pro_emb_batch", "value"):
                assert torch.equal(getattr(reference, name), getattr(batch, name).cpu()), name
            assert reference.num_graphs == batch.num_graphs


def _train_on_cuda(monkeypatch, production, noise=False, scale=1.0):
    """4 epochs of 3 batches. Returns the per-step losses and the final validation MSE."""
    proteins = FakeProteins()
    train, val = make_rows(20, seed=1), make_rows(11, seed=2)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = production
    losses = []

    class RecordingMSE(torch.nn.MSELoss):
        def forward(self, prediction, target):
            loss = super().forward(prediction, target)
            losses.append(loss.item())
            return loss

    monkeypatch.setattr(torch.nn, "MSELoss", RecordingMSE)
    torch.manual_seed(3)
    model = build_model("kcat", "MACCSKeys", 1024, 64, PROTEIN_DIM, 3, 0.5, 3)
    if noise:
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in model.parameters():
                p += NOISE_U * p.abs() * torch.randn(p.shape, generator=generator)
    with torch.no_grad():
        for p in model.parameters():
            p *= scale
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader, val_loader = wrapper_loader(train, True, proteins, "cuda"), wrapper_loader(val, False, proteins, "cuda")
    torch.manual_seed(5)
    for _ in range(4):
        trainer.train_one_epoch(model, train_loader, optimizer, device)
    train_losses = losses[:]
    val_mse = trainer.evaluate_loader(model, val_loader, device, show_progress=False)[2]["loss"]
    return train_losses, val_mse


K = 10  # the production gap must stay within 10x the production-path noise (perturbed weights) gap
NEGATIVE_CONTROL_SCALE = 1.05 if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1" else 1.0  # scales only the production run's initial weights
NOISE_U = 2**-11  # one rounding of the weights at the production precision (TF32 / fp16 unit roundoff)


@pytest.mark.gpu
@needs_cuda
def test_production_flags_within_fp32_noise_control(monkeypatch):
    """Production flags (TF32 matmul and cudnn, cudnn.benchmark; training is fp32 without autocast) against
    deterministic fp32 with TF32 and benchmark off, on the same seed and batches.

    A noise control reruns the production path with the initial weights perturbed by 2**-11 (relative, one TF32 rounding).
    Floor = |perturbed production - production|, gap = |production - fp32|. Per-step loss gap and final validation MSE gap
    must stay within K times the floor. Step 0 is judged like every other step.
    """
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    configs = [(False, False, 1.0), (True, True, 1.0), (True, False, 1.0)]
    if NEGATIVE_CONTROL_SCALE != 1.0:
        configs.append((True, False, NEGATIVE_CONTROL_SCALE))  # the gap uses this run; the floor keeps the unscaled production run
    try:
        runs = [_train_on_cuda(monkeypatch, *config) for config in configs]
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = flags
    (ref, ref_mse), (noise, noise_mse), (clean, clean_mse), (prod, prod_mse) = runs[0], runs[1], runs[2], runs[-1]
    assert len(ref) == len(noise) == len(prod) == 12
    ref, noise, clean, prod = (torch.tensor(x, dtype=torch.float64) for x in (ref, noise, clean, prod))
    tiny = 1e-7 * ref.abs().max()
    gap, floor = (prod - ref).abs(), (noise - clean).abs()
    final_gap, final_floor = abs(prod_mse - ref_mse), abs(noise_mse - clean_mse)
    tiny0 = 1e-7 * ref[0].abs()
    gap0, floor0 = float(gap[0]), float(floor[0])
    print(
        f"eitlem tf32: max gap={gap.max():.3e}, max floor={floor.max():.3e}, "
        f"ratio={gap.max() / max(floor.max(), tiny):.2f}, K={K}, step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}, "
        f"final gap={final_gap:.3e}, final floor={final_floor:.3e}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap.max() <= K * max(floor.max(), tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * abs(ref_mse))
