# emulator_bench

Multi-seed evaluation of EITLEM-Kinetics with its original settings on the default EMULaToR splits.
The wrapper reuses the original model code in `Code/` (`KCM.py`, `KMP.py`). It tunes nothing.

## Tasks and data

- Tasks: `kcat` and `km`. Both live in `data/processed/datasets/kinetic_dataset/<task>`.
- Splits per task: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`,
  `enzyme_sequence_splits/threshold_<t>`, `enzyme_structure_splits/threshold_<t>`, `substrate_splits/threshold_<t>`, `conformer_cosine_splits/threshold_<t>`. The wrapper finds the thresholds on disk.
- Columns used: `sequence`, `smiles`, `log10_value`.

## Model and settings

- Protein input: per-residue ESM1v (`esm1v_t33_650M_UR90S_1`, layer 33, 1280-d) in fp32.
  Sequences longer than 1022 residues use a sliding window (the original data had none).
- Molecule input: MACCSKeys fingerprint (167 bits).
- Training: AdamW, lr 1e-3, weight decay 0.01 (PyTorch default), batch size 200, 100 epochs,
  MultiStepLR (milestones 50 and 80, gamma 0.9), MSE loss, fp32 by default (`--amp` switches to bf16, or fp16 on pre-Ampere GPUs), no early stopping.
- The model of the last epoch is evaluated, as in the original code.
- Seeds: 666, 777, 888.

See `REPORT.md` for sources and details.

## Files

- `cache_embeddings.py`: builds the ESM1v and fingerprint caches for all splits.
- `train_single_target_tvt.py`: trains one split and one seed. It writes `checkpoint.pth` after every epoch and resumes from it (see Resume).
- `run_split_benchmarks.py`: caches features, then trains every task, split and seed. `--gpus 0 1` runs one training per GPU. `--runs_per_gpu` defaults to 1. Two runs on one A10 (24 GB) risk out-of-memory: one run peaked near 19 GB, and long-sequence batches can raise it. The training script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (if unset), which cut the peak to about 15 GB on a test subset. `--amp` turns on mixed precision. It skips finished runs (`final_results_test.csv` exists) and resumes unfinished ones. `--overwrite` retrains everything from epoch 1.
  `--num_workers` (default 8) sets the threads that build batches. `--prefetch_factor` (default 2) sets how many batches they build ahead. Each batch in flight holds GPU memory.
- `esm_deps.py`: makes fair-esm importable (see Run).
- `aggregate_eitlem_results.py`: mean and std over seeds.
- `predict_single_target.py`: inference from a saved `lastmodel.pth`.

## Run

Environment: `mldb`. The wrapper unpacks the fair-esm 2.0.0 wheel (sha256 checked) to `<output_dir>/pydeps` when the env has another `esm` package (`esm_deps.py`). It does not change the env.
Outputs go to `<EMULaToR>/outputs/baselines/EITLEM-Kinetics` (override with `--output_dir`).
See `commands.txt` in the parent folder for the exact commands. Run them from `baselines/EITLEM-Kinetics`.

## Resume

After each epoch the training script writes `checkpoint.pth` (atomic) in the run folder. It holds model, optimizer, scheduler and GradScaler state, the epoch, the `logfile.csv` rows, and the python, numpy, torch CPU and CUDA RNG states. The shuffle order comes from the torch RNG, so it resumes too.
If `checkpoint.pth` exists at start, the script restores it, rewrites `logfile.csv` to the checkpointed rows and continues at the next epoch. It stops with an error if task, split paths, seed, epochs, batch size, lr, precision, fingerprint or predictor type differ. `--no_resume` ignores the checkpoint and starts at epoch 1 with a new `logfile.csv`.
The script deletes `checkpoint.pth` after `run_summary.json` is written. A resumed run is exact on CPU with deterministic settings. On GPU, nondeterministic kernels (and `cudnn.benchmark`) can make the numbers differ slightly.

## Verify equivalence

```
emulator_bench/verify_equivalence.sh          # CPU, fp32, about 10 s
emulator_bench/verify_equivalence.sh --gpu    # also the CUDA tests; needs one CUDA device
emulator_bench/verify_equivalence.sh --negative-control  # GPU precision test must FAIL
```

Expected env: `mldb` (torch, torch_geometric, rdkit, pandas, pytest, Biohub `esm`). Never `pip install fair-esm`: it overwrites the Biohub `esm` package. `esm_deps.py` (`setup_esm`) unpacks the sha256-checked fair-esm 2.0.0 wheel from PyPI into `<out_root>/pydeps` and puts it first on `sys.path`. It never touches site-packages. The tests reach fair-esm only this way, through `emulator_bench.feature_pipeline`. The first run needs network once unless `<out_root>/pydeps` exists. The script does not create envs. It fails if any CPU test is skipped. It needs no private data: it builds synthetic proteins and a tiny random ESM1v-shaped model.

What `emulator_bench/tests/test_original_equivalence.py` compares:
- Batches: `EitlemDataLoader` against the original PyG `DataLoader` (shuffle, persistent workers), over three epochs. Batches must be identical.
- Training: `fit` against the original `Trainer` and `Tester` loop, for kcat and km, with the original model classes at hidden size 64 and 3 layers. Losses and final weights within rtol 1e-5.
- ESM1v features, split batches: 40 sequences of 300-900 residues, so the length-sorted batching at 12000 residues makes several batches. Same tolerance.
- ESM1v features, long protein: a 1900-residue protein embedded in 1022-residue windows at stride 800 equals the average of the original per-window features. Same tolerance.
- ESM1v features: the cache builder and memory-mapped store against `Code/extract.py`, on a random ESM1v-shaped network (33 layers, 16-d). Features within rtol 1e-4. Add `-m esm_weights` to run the same check with the real weights; it skips if `$TORCH_HOME/hub/checkpoints` has none.
- `--gpu`, loader: the side-stream copies give the CPU loader batches, in the same order.
- `--gpu`, precision: the production flags (TF32 and `cudnn.benchmark`) against deterministic fp32, same seed and batches. A noise control reruns the production path with initial weights perturbed by 2**-11 (relative, one TF32 rounding). The per-step loss gap and the final validation MSE gap between production and fp32 must be within 10 times the gap the perturbation causes. Step 0 is judged like every other step: its gap must be at most 10 times its floor. The test prints the measured ratio (run with `-s`).
- `--negative-control`: reruns only that test with the production run's initial weights multiplied by 1.05 (`EQUIV_NEGATIVE_CONTROL=1`). The test must fail. The script exits 0 only then.

Production training is fp32 with TF32. `--amp` is not used for the runs: it was slower. The tests use a smaller model and a random encoder, so they do not check the real ESM1v weights or the full model size. TF32 differs from fp32 by rounding.

## Output layout

- `embeddings/proteins_v3/`, `embeddings/ligands_v3/`, `embeddings/manifest.json`: feature cache.
  `proteins_v3/shard_<n>.npy` is one uncompressed float32 array with the residues of many proteins. `shard_<n>.json` maps each protein key to `[offset, length]`.
  Training reads the shards with `np.load(mmap_mode="r")`, so all runs share the OS page cache.
  A shard holds about 1M residues (5 GB). The shard `.json` is written last, so an interrupted build resumes and skips finished shards.
  Protein keys cover the ESM model, layer, dtype, window size and stride. Ligand keys cover fingerprint type, radius and bits.
  Change `CACHE_VERSION` in `common.py` to force a rebuild. Caches from version 2 (`proteins_v2/`, one `.npz` per protein) are not read. Rebuild once and delete the old folders.
- `<task>/<split_group>/<split_name>/seed_<n>/`: `lastmodel.pth`, `logfile.csv`, `final_results_{train,val,test}.csv`,
  `pred_label_{train,val,test}.csv`, `run_summary.json`; `checkpoint.pth` while the run is unfinished. `<split_name>` is `all` for flat split groups.
- `aggregate_eitlem_results.csv`: mean and std over seeds.

## Host setup (A100 host `e2ea100_2`)

The runs use a conda env `eitlem` with the versions of the local `mldb` env.
The torch build is `2.12.0+cu130` (mldb has `2.12.0+cu132`), because the host driver (580.95.05) supports CUDA 13.0.
fair-esm is unpacked by `esm_deps.py` under `<output_dir>/pydeps`.

```
~/miniconda3/bin/conda create -y -n eitlem python=3.12
E=~/miniconda3/envs/eitlem/bin
$E/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
$E/pip install torch_geometric==2.8.0 rdkit==2026.3.3 numpy==2.4.4 pandas==2.3.3 scikit-learn==1.8.0 \
  scipy==1.17.1 pyarrow==23.0.1 rich==14.3.3 tqdm==4.68.2 xxhash==3.7.0
```
