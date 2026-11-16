import argparse
import random
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
from emulator_bench.dataset import EitlemDataLoader  # noqa: E402

EPOCHS = 4

torch.use_deterministic_algorithms(True)


class FakeProteins:
    def get(self, sequence):
        return np.random.default_rng(len(sequence)).random((len(sequence), 5)).astype(np.float32)


class FakeLigands:
    def get(self, smiles):
        return {"fingerprint": np.random.default_rng(int(smiles[1:])).random(8).astype(np.float32)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(13, 16), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(16, 1))

    def forward(self, batch):
        pooled = torch.zeros(batch.num_graphs, 5).index_add_(0, batch.pro_emb_batch, batch.pro_emb)
        return self.net(torch.cat([batch.x, pooled], dim=1)).squeeze(-1)


def make_args(**overrides):
    values = dict(
        task_name="t",
        train_path="a",
        val_path="b",
        test_path="c",
        seed=1,
        epochs=EPOCHS,
        batch_size=7,
        lr=1e-2,
        amp=False,
        predictor_type="kcat",
        mol_type="MACCSKeys",
        radius=4,
        nbits=1024,
    )
    return argparse.Namespace(**{**values, **overrides})


def run(args, out_dir, resume=True):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    loaders = {}
    for name, n in (("train", 40), ("val", 17)):
        frame = pd.DataFrame(
            {
                "sequence": ["ACDE" * (i % 5 + 1) for i in range(n)],
                "smiles": [f"S{i}" for i in range(n)],
                "log10_value": np.random.default_rng(n).random(n),
            }
        )
        loaders[name] = EitlemDataLoader(frame, FakeProteins(), FakeLigands(), device, args.batch_size, shuffle=name == "train", num_workers=2)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=0.9)
    trainer.fit(args, model, optimizer, scheduler, None, loaders, device, None, out_dir, resume=resume)
    return model


def kill_after(monkeypatch, epoch):
    """Raises right after the checkpoint of `epoch` is written, like a kill between epochs."""
    save = trainer.save_checkpoint

    def save_then_die(path, **state):
        save(path, **state)
        if state["epoch"] == epoch:
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "save_checkpoint", save_then_die)


def interrupted_run(monkeypatch, out_dir, epoch=2):
    with monkeypatch.context() as patch:
        kill_after(patch, epoch)
        with pytest.raises(KeyboardInterrupt):
            run(make_args(), out_dir)


def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_resume_matches_straight_run(tmp_path, monkeypatch):
    straight = run(make_args(), tmp_path / "straight")
    interrupted_run(monkeypatch, tmp_path / "split")
    resumed = run(make_args(), tmp_path / "split")
    assert torch.equal(weights(straight), weights(resumed))
    drop = ["elapsed_seconds"]
    a = pd.read_csv(tmp_path / "straight" / "logfile.csv").drop(columns=drop)
    b = pd.read_csv(tmp_path / "split" / "logfile.csv").drop(columns=drop)
    pd.testing.assert_frame_equal(a, b)
    assert list(b["epoch"]) == list(range(1, EPOCHS + 1))


def test_logfile_truncated_to_checkpoint(tmp_path, monkeypatch):
    interrupted_run(monkeypatch, tmp_path)
    with open(tmp_path / "logfile.csv", "a") as handle:
        handle.write("3,0.1,0,0,0,0,0,0\n")  # row of an epoch that never reached its checkpoint
    run(make_args(), tmp_path)
    assert list(pd.read_csv(tmp_path / "logfile.csv")["epoch"]) == list(range(1, EPOCHS + 1))


def test_no_resume_starts_fresh(tmp_path, monkeypatch):
    interrupted_run(monkeypatch, tmp_path)
    run(make_args(), tmp_path, resume=False)
    assert list(pd.read_csv(tmp_path / "logfile.csv")["epoch"]) == list(range(1, EPOCHS + 1))


@pytest.mark.parametrize(
    "change", [{"seed": 2}, {"lr": 1e-3}, {"batch_size": 8}, {"amp": True}, {"epochs": 9}, {"train_path": "x"}, {"task_name": "u"}]
)
def test_resume_refuses_changed_args(tmp_path, monkeypatch, change):
    interrupted_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="arguments differ"):
        run(make_args(**change), tmp_path)
