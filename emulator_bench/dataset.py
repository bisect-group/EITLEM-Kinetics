import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import LigandEmbeddingStore, ProteinEmbeddingStore


@dataclass
class Batch:
    """Residues of all proteins concatenated; `pro_emb_batch` maps each residue to its row."""

    x: torch.Tensor
    pro_emb: torch.Tensor
    pro_emb_batch: torch.Tensor
    value: Optional[torch.Tensor]
    num_graphs: int


class EitlemDataLoader:
    """Builds batches in threads straight from the memory-mapped protein cache.

    On CUDA each thread fills a pinned buffer and copies it to the GPU on a side stream, so the copy
    overlaps the running step. `prefetch_factor` batches are in flight; each holds GPU memory.
    The sampler seeds follow `torch.utils.data.DataLoader` with persistent workers.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandEmbeddingStore,
        device: torch.device,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: Optional[str] = "log10_value",
    ):
        sequences = [normalize_sequence(str(value)) for value in frame[sequence_col]]
        proteins = {sequence: protein_store.get(sequence) for sequence in set(sequences)}
        self.proteins = [proteins[sequence] for sequence in sequences]
        self.dim = next(iter(proteins.values())).shape[1]
        self.fingerprints = np.stack([ligand_store.get(str(smiles))["fingerprint"] for smiles in frame[smiles_col]])
        self.values = None
        if target_col is not None and target_col in frame.columns:
            self.values = torch.tensor(frame[target_col].to_numpy(dtype=np.float32))
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.pool = ThreadPoolExecutor(num_workers)
        self.seeded = False

    def __len__(self) -> int:
        return math.ceil(len(self.proteins) / self.batch_size)

    def _build(self, indices: list):
        with torch.cuda.stream(self.stream) if self.cuda else nullcontext():
            lengths = [len(self.proteins[i]) for i in indices]
            pro_emb = torch.empty((sum(lengths), self.dim), pin_memory=self.cuda)
            target, offset = pro_emb.numpy(), 0
            for i, length in zip(indices, lengths):
                target[offset : offset + length] = self.proteins[i]
                offset += length
            pro_emb_batch = torch.repeat_interleave(torch.arange(len(indices)), torch.tensor(lengths))
            x = torch.from_numpy(self.fingerprints[indices])
            value = None if self.values is None else self.values[indices]
            batch = Batch(
                x=x.to(self.device, non_blocking=True).float(),
                pro_emb=pro_emb.to(self.device, non_blocking=True),
                pro_emb_batch=pro_emb_batch.to(self.device, non_blocking=True),
                value=None if value is None else value.to(self.device, non_blocking=True),
                num_graphs=len(indices),
            )
            if not self.cuda:
                return batch, None
            event = torch.cuda.Event()
            event.record(self.stream)
            return batch, event

    def __iter__(self) -> Iterator[Batch]:
        if not self.seeded:  # DataLoader draws its base seed once, when the persistent workers start
            torch.empty((), dtype=torch.int64).random_()
            self.seeded = True
        sampler = RandomSampler(range(len(self.proteins))) if self.shuffle else SequentialSampler(range(len(self.proteins)))
        batches = iter(BatchSampler(sampler, self.batch_size, drop_last=False))
        pending = deque()
        for indices in batches:
            pending.append(self.pool.submit(self._build, indices))
            if len(pending) > self.prefetch_factor:
                yield self._ready(pending.popleft())
        while pending:
            yield self._ready(pending.popleft())

    def _ready(self, future) -> Batch:
        batch, event = future.result()
        if event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(event)
            for tensor in (batch.x, batch.pro_emb, batch.pro_emb_batch, batch.value):
                if tensor is not None:
                    tensor.record_stream(current)
        return batch
