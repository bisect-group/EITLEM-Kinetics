# emulator_bench

Multi-seed evaluation of EITLEM-Kinetics with its original settings on the default EMULaToR splits.
The wrapper reuses the original model code in `Code/` (`KCM.py`, `KMP.py`). It tunes nothing.

## Tasks and data

- Tasks: `kcat` and `km`. Both live in `data/processed/datasets/kinetic_dataset/<task>`.
- Splits per task: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`,
  `enzyme_sequence_splits/threshold_<t>`, `substrate_splits/threshold_<t>`. The wrapper finds the thresholds on disk.
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
- `train_single_target_tvt.py`: trains one split and one seed.
- `run_split_benchmarks.py`: caches features, then trains every task, split and seed. `--gpus 0 1 --runs_per_gpu 2` runs several trainings at once, one GPU each. `--amp` turns on mixed precision.
- `aggregate_eitlem_results.py`: mean and std over seeds.
- `predict_single_target.py`: inference from a saved `lastmodel.pth`.

## Run

Environment: `mldb` with `fair-esm` (the `esm.pretrained.esm1v_t33_650M_UR90S_1` loader).
Outputs go to `<EMULaToR>/outputs/baselines/EITLEM-Kinetics` (override with `--output_dir`).
See `commands.txt` in the parent folder for the exact commands. Run them from `baselines/EITLEM-Kinetics`.

## Output layout

- `embeddings/proteins_v2/`, `embeddings/ligands_v2/`, `embeddings/manifest.json`: feature cache.
  Protein keys cover the ESM model, layer, dtype, window size and stride. Ligand keys cover fingerprint type, radius and bits.
  Change `CACHE_VERSION` in `common.py` to force a rebuild.
- `<task>/<split_group>/<split_name>/seed_<n>/`: `lastmodel.pth`, `logfile.csv`, `final_results_{train,val,test}.csv`,
  `pred_label_{train,val,test}.csv`, `run_summary.json`. `<split_name>` is `all` for flat split groups.
- `aggregate_eitlem_results.csv`: mean and std over seeds.
