# Tools and script organization

Use `python main.py run --config ...` for configured experiments. Search implementation modules live in `flashnas/search/`; standalone utilities are grouped by purpose under `tools/` and use descriptive `snake_case` filenames.

## Filename mapping

| Previous root filename | New location |
| --- | --- |
| `0-rm_compiled.py` | `tools/maintenance/clean_python_cache.py` |
| `1-cal_max_parm.py` | `tools/analysis/estimate_parameters.py` |
| `1-search_draw_parato.py` | `flashnas/search/plot_pareto.py` |
| `1-search_first_stage.py` | `flashnas/search/stage_one.py` |
| `1-search_second_stage.py` | `flashnas/search/stage_two.py` |
| `2-gen_parm_flops_npy.py` | `tools/data/generate_predictor_features.py` |
| `2-merge_repeat_arch_log.py` | `tools/data/merge_training_runs.py` |
| `2-merge_repeat_arch_txt.py` | `tools/data/merge_architecture_lists.py` |
| `2-read_np.py` | `tools/data/inspect_predictor_arrays.py` |
| `2-rm_error_file.py` | `tools/maintenance/prune_incomplete_runs.py` |
| `3-cal_acc_parm_op.py` | `tools/analysis/profile_architectures.py` |
| `3-generate.py` | `tools/analysis/inspect_predictor_outputs.py` |

The configured experiment commands and shell launchers already reference the new locations. For direct utility execution, run the module from the repository root. No project installation is required.

## Data tools

```bash
python -m tools.data.inspect_predictor_arrays --data-dir 2-npy/imagenet100
python -m tools.data.merge_architecture_lists --inputs first.txt second.txt --output combined.txt
python -m tools.data.merge_training_runs --sources runs-a runs-b --output combined-runs
python -m tools.data.generate_predictor_features \
  --dataset imagenet100 --architectures architectures.txt \
  --logs training-runs --prefix BNN --epochs 25 --output 2-npy/imagenet100
```

Architecture text records contain three lines per architecture: channels, kernel types, and strides. This legacy text order differs from `ARCH_ALL` (channels, strides, kernel types). Feature generation pairs record N with `BNN<N>/log.txt`, using the specified recorded labeling budget. Input records and run IDs must correspond. The generator writes only complete measured labels with aligned features and does not add a built-in baseline accuracy.

Text merging deduplicates architectures. Run-directory merging copies and renumbers all runs and records a `source-map.txt`. They are separate operations: use the map to maintain the architecture-to-log correspondence when preparing predictor labels. Existing output files/directories are protected from overwrite.

## Analysis tools

```bash
python -m tools.analysis.estimate_parameters --channels 32 64 128
python -m tools.analysis.profile_architectures --help
python -m tools.analysis.inspect_predictor_outputs --help
python -m flashnas.search.plot_pareto --help
```

`estimate_parameters` preserves the original simple 3×3 plus two 1×1 block-chain calculation. Use candidate cost models for binary-network OP/parameter analysis. `profile_architectures` supports legacy example candidates or `--read_txt 1 --architecture_file <path>`. `inspect_predictor_outputs` compares checkpoint predictions with held-out labels using `--data_root` and `--out_path`.

## Maintenance tools

```bash
# Preview cache directories
python -m tools.maintenance.clean_python_cache --root .
# Remove the listed caches
python -m tools.maintenance.clean_python_cache --root . --apply

# Preview legacy BNN runs whose existing log lacks a Best Acc record
python -m tools.maintenance.prune_incomplete_runs --root training-runs
# Delete the listed incomplete runs
python -m tools.maintenance.prune_incomplete_runs --root training-runs --apply
```

Maintenance commands have no import-time side effects. Missing logs are left for manual inspection. Explicit roots replace personal machine paths; cleanup uses Python filesystem operations and skips symlink run directories.
