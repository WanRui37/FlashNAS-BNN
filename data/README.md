# Data contract

Keep licensed images and generated arrays outside Git. `configs/` contains the input locations. Training configuration paths are relative to `train_imagenet/`; predictor/search paths are relative to the repository root.

## ImageNet-100

Use `data/imagenet100/train/<class>/` and `data/imagenet100/val/<class>/` with the same 100 class names and disjoint image splits. The exact published subset/class mapping is not bundled. Keep your class list and subset-generation procedure with your experiment; an arbitrary 100-class subset is a new experiment.

## Predictor arrays

Store files under `2-npy/<dataset>/` or override `parameters.data_root`:

| File | Shape | Meaning |
| --- | --- | --- |
| `2-np_best_accuracy.npy` | `(N,)` or `(N, 1)` | Measured short-training validation accuracy in percent |
| `2-np_feature_single_vec.npy` | `(N, L, 8)` | Per-layer channel/kernel encoding and FP/Binary operation/parameter features |
| `2-np_feature_total_vec.npy` | `(N, 11)` for ImageNet-100 | Global operation/parameter totals, depth and stage-width features |

For ImageNet-100, `L=32`; CIFAR predictor datasets use `L=25`. Rows must refer to the same architecture across all three arrays. Existing feature encoders and normalization are in `utils/calculate.py` and `utils/dataset.py`; preserve feature order and units when supplying checkpoints. Labels must come from your recorded experiments. The code scales percentage labels by 100 for predictor fitting.

A single seeded permutation splits the rows into disjoint training and validation subsets. This release fixes the legacy overlapping-split bug; refitting can therefore produce different predictor results. It does not retroactively validate previously reported predictor metrics.

## Candidate encoding

`ARCH_ALL` concatenates equally sized arrays: `(channels..., strides..., kernel_types...)`. Kernel type 0 means 3×3 and 1 means 1×1. Search checkpoints pad unused positions with -1; the model builder removes padding. Strides are 1 or 2. The configs retain the architecture from the original ImageNet-100 shell example.

Models read this configuration at construction. The framework reloads the relevant original module for each timing candidate, so one candidate cannot silently reuse the previous candidate's architecture. Constructing models is intended for sequential experiment execution, not concurrent threads mutating process environment.
