# EITLEM-Kinetics emulator_bench report

## Method summary
EITLEM-Kinetics (Shen et al., Chem Catalysis 4, 101094, 2024) predicts kcat and Km of enzyme mutants.
Protein input is per-residue ESM1v (esm1v_t33_650M_UR90S_1, layer 33, 1280-d), with no pooling.
Substrate input is a MACCSKeys fingerprint (167 bits) from SMILES.
The model (ProMolAtt + AttentionAgg + FFN) fuses both inputs and outputs log10 of the value.
The paper trains kcat and Km base models, then runs 8 iterations of ensemble transfer learning with KKm data.
The wrapper trains only the direct (base) kcat and Km models. KKm data does not exist in EMULaToR.

## Original default settings
| Setting | Value | Source |
|---|---|---|
| Hidden / protein dim / layers / att layers | 512 / 1280 / 10 / 10 | Code/iter_train_scripts.py:36-41 |
| Dropout | 0.5 | iter_train_scripts.py:36-41 |
| Optimiser | AdamW, lr 1e-3, weight decay 0.01 (PyTorch default) | iter_train_scripts.py:81 |
| Schedule | MultiStepLR, milestones 50 and 80, gamma 0.9 | iter_train_scripts.py:82 |
| Epochs | 100 (base model, no pre-trained weights) | iter_train_scripts.py:26 |
| Batch size | 200, shuffle, drop_last False | iter_train_scripts.py:69 |
| Loss | MSE on log10 value | iter_train_scripts.py:84, log10 default :180 |
| Precision | fp32 (autocast and GradScaler are commented out) | Code/eitlem_utils.py:55-67 |
| Early stopping, grad clip | none | iter_train_scripts.py |
| Checkpoint | weights of the last epoch (save_model deletes the earlier file) | eitlem_utils.py:47-51 |
| Features | ESM1v layer 33 fp32, MACCSKeys | Code/extract_protein.py, paper p.3 |
| Seeds | none set in the code; paper states none. Wrapper uses 666 777 888 | - |

Paper vs code: the main PDF refers to Table S2 for training parameters, but the supplement is not in the PDF.
So the code values above are the reference. The old wrapper JSON had weight decay 0.0. The original code uses the AdamW default 0.01.

## What the wrapper does now
`run_split_benchmarks.py` builds the feature cache, then trains each task (kcat, km) x split x seed.
Splits are found on disk: random_splits_grouped_sequence, random_splits_grouped_smiles, uniprot_time_splits,
enzyme_sequence_splits/threshold_*, substrate_splits/threshold_*.
Each run trains 100 epochs, logs val metrics per epoch, and evaluates the final-epoch model on train, val and test.
Outputs: `<output_dir>/<task>/results/<split_group>/<split_name>/seed_<n>/`.
Default output dir is `EMULaToR/outputs/baselines/EITLEM-Kinetics` (gitignored). Nothing is written under `data/`.
`aggregate_eitlem_results.py` gives mean and std over seeds.

## Files removed
- emulator_bench: tune_optuna.py, launch_parallel_optuna.py, launch_parallel_retrain_from_optuna.py,
  default_hparams_kcat_original.json, default_hparams_km_original.json.
- Submodule root: run_optuna_km_sweeps.sh (Optuna sweep script).

## Files changed
- common.py: repo-relative data and output defaults, seeds 666 777 888, simpler split discovery, removed unused helpers.
- train_single_target_tvt.py: rewritten to the original protocol (see below).
- run_split_benchmarks.py: task x split x seed loop only; no hyperparameter flags.
- cache_embeddings.py, feature_pipeline.py: fp32 ESM1v and fp32 cache, no autocast; multi-task discovery.
- modeling.py: kcat and km only. The kkm ensemble is removed.
- predict_single_target.py: uses `final_model.pth`, fp32.
- aggregate_eitlem_results.py: rewritten for the new layout.
- README.md (emulator_bench) and commands.txt (submodule root): rewritten, no Optuna, no foreign home paths.

## Bugs fixed
- Hardcoded `/home/mallikarjun` and `/home/da24s023` paths and non-existent `data/processed/baselines` defaults.
- Default split groups lacked `uniprot_time_splits`; included absent `group_shuffle_splits`.
- Aggregator read only `retrain_from_optuna` directories.
- Results were written next to the split files under `data/`.
- Old training used cosine schedule, warmup, grad clip, patience 20, best-val checkpoint, bf16, weight decay 1e-4 and batch size 128. All are removed.

## Deviations from the original (and why behaviour is unchanged)
- Cached features instead of recomputing ESM1v and RDKit per sample. Same values (fp32 cache). Padded batches in ESM can differ by float rounding.
- Sequences longer than 1022 residues use a sliding window with averaged overlap. The original removed such sequences. EMULaToR keeps them (about 2.7% of rows).
- Dataset read from parquet instead of the original pair-info files. Model code in `Code/` is unchanged.
- Validation runs every epoch for logging only. It does not affect training or model choice.
- Original code and Code/ files were not edited.

## Commands
```
cd baselines/EITLEM-Kinetics
python emulator_bench/run_split_benchmarks.py --tasks kcat km --seeds 666 777 888 --device cuda:0
python emulator_bench/aggregate_eitlem_results.py
```
Needs an env with fair-esm (`esm.pretrained.esm1v_t33_650M_UR90S_1`), torch_geometric and rdkit.

## Smoke test
Env: mldb, plus fair-esm 2.0.0 in a scratch dir via PYTHONPATH (mldb has the Biohub `esm`, which has no esm1v loader). Weights via TORCH_HOME in scratch.
Data: 64 rows each of kcat random_splits_grouped_sequence and enzyme_sequence_splits/threshold_0.01 (48/8/8), GPU 1, 1 epoch, seed 666.
Result: cache build, training, final evaluation, aggregation and predict_single_target all ran. Outputs are in
`scratchpad/bench_smoke/EITLEM-Kinetics`. Metrics are meaningless after 1 epoch.
A full 100-epoch run was not done.

## Changes after review

- `train_single_target_tvt.py` sets `torch.backends.cudnn.allow_tf32 = True` and `torch.backends.cuda.matmul.allow_tf32 = True`, as in `Code/iter_train_scripts.py:186-187`. The cache and model stay fp32 and there is no autocast. TF32 only changes matmul and conv rounding on Ampere and later GPUs.
- Re-smoke (kcat, random grouped sequence, seed 666, 1 epoch, GPU 1): training and final evaluation ran. Output: `bench_smoke/EITLEM-Kinetics/out2`.

## Open issues
- mldb has no ESM1v loader. Install fair-esm in a separate env for real runs. Cache size is about 70 GB (fp32) for kcat and km together.
- The paper's final model uses 8 rounds of iterative transfer learning with KKm data. The wrapper tests the base model only. KKm data is not in EMULaToR.
- Model choice is the last epoch (original code), not best validation. Change only if the paper wants val-based selection.
- Training 100 epochs with per-sample npz loading may be slow on the large splits. Use more `--num_workers`.
- Weight decay 0.01 follows the code default (not set explicitly in the original code).
- `torch.backends.cudnn.benchmark = True` is not in the original code. It is speed-only and can make runs slightly non-deterministic.
