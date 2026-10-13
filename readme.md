# FlashNAS-BNN

**Chain-Structured Binary Neural Architecture Search for Efficient Visual Recognition across Platforms**

FlashNAS-BNN searches chain-structured binary neural networks using an accuracy predictor (CIF-Net), evolutionary candidate search, and hardware-aware candidate selection. This repository contains predictor training, two-stage search, and ImageNet-100 candidate training/evaluation code.

## Highlights

- **Two-stage architecture search:** CIF-Net-guided evolutionary exploration followed by hardware-aware candidate selection.
- **Configurable experiments:** eight JSON recipes cover predictor fitting, architecture search, network training, evaluation, and Pareto visualization.
- **Traceable runs:** resolved parameters, commands, Git revision, execution status, and console logs are saved for each experiment.
- **Candidate-specific timing:** construct each searched binary network and measure its PyTorch inference latency on CPU or CUDA.
- **Reproducible development:** disjoint seeded predictor splits, CPU regression tests, and GitHub Actions checks.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data and checkpoints](#data-and-checkpoints)
- [Training and evaluation](#training-and-evaluation)
- [Architecture search](#architecture-search)
- [Reported results](#reported-results)
- [Project structure](#project-structure)
- [Reproducibility and development](#reproducibility-and-development)

## Installation

The reference environment uses Python 3.10, PyTorch 2.0.0, torchvision 0.15.0, and CUDA 11.7. [environment.yml](environment.yml) preserves the original full environment export. For a compact environment, run from the repository root:

```bash
git clone https://github.com/WanRui37/FlashNAS-BNN.git
cd FlashNAS-BNN
conda env create -f environment-minimal.yml
conda activate flashnas-bnn
python -m pip install -e . --no-deps
python -m flashnas doctor
```

Alternatively, create a Python 3.10 environment, install a matching PyTorch/torchvision build using the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/), then install the research dependencies and launcher:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

CPU execution is suitable for framework checks, predictor experiments, and local timing experiments. The existing image-training loop requires an NVIDIA GPU and NVIDIA Apex. Install Apex in a separate directory using the pinned revision:

```bash
git clone --branch 24.04.01 https://github.com/NVIDIA/apex.git /tmp/flashnas-apex
python -m pip install -v --disable-pip-version-check --no-build-isolation /tmp/flashnas-apex
```

GCN predictor baselines additionally use `python -m pip install -r requirements-graph.txt`. The primary `OUR` predictor uses the standard research dependencies. Run `doctor` to inspect the installed modules.

## Quick start

Inspect an experiment without downloading data or starting GPU work:

```bash
python -m flashnas run --config configs/search-stage1-imagenet100.json --dry-run
bash run.sh run --config configs/train-stage1-imagenet100.json --dry-run
```

Every configuration is ordinary JSON. `--set` overrides an existing value; it rejects misspelled keys:

```bash
python -m flashnas run --config configs/search-stage1-imagenet100.json \
  --set parameters.seed=42 \
  --set parameters.logdir=./search_output/imagenet100/seed42/stage1 \
  --set run_dir=./outputs/runs/search-seed42 --dry-run
```

Remove `--dry-run` after preparing the required inputs. Each execution saves the resolved configuration, exact command, Git revision, dirty-worktree indicator, timestamps, exit code, and console output in `run_dir`. Existing run manifests are protected from overwrite. Use different `run_dir` **and** output paths for independent runs. Relative parameter paths are resolved from the underlying script's working directory, shown by `--dry-run`.

## Data and checkpoints

ImageNet-100 uses an ImageFolder layout with the same 100 class names in both splits:

```text
data/imagenet100/
├── train/<class-name>/<image files>
└── val/<class-name>/<image files>
```

Place your ImageNet-100 training and validation images in this layout. See [data preparation](data/README.md) for predictor-array shapes and architecture encoding.

CIF-Net training requires aligned, measured architecture labels and features in:

```text
2-npy/imagenet100/
├── 2-np_best_accuracy.npy
├── 2-np_feature_single_vec.npy
└── 2-np_feature_total_vec.npy
```

Labels use accuracy percentages, such as 81.40. Feature encoders and normalization are shared by predictor fitting and search.

See [checkpoint formats and locations](checkpoints/README.md) to organize CIF-Net, CLE, candidate-search, and trained-network artifacts. Predictor training and two-stage network training generate checkpoints for subsequent search and evaluation.

## Training and evaluation

The training recipes include an explicit candidate architecture. Set `environment.ARCH_ALL` to train your selected architecture, using the **same** value for both stages. Architecture encoding is documented in [data/README.md](data/README.md).

```bash
# 1. Candidate training, stage 1: 128 epochs
python -m flashnas run --config configs/train-stage1-imagenet100.json

# 2. Candidate training, stage 2: initialize from stage-1 checkpoint
python -m flashnas run --config configs/train-stage2-imagenet100.json

# 3. Evaluate trained weights on the validation split
python -m flashnas run --config configs/evaluate-imagenet100.json
```

The examples use Adam, learning rate 1.25e-3, batch size 32 per process, seed 2024, and one GPU. Stage 1 uses weight decay 1e-5; stage 2 uses 0. Set `nproc` and `environment.CUDA_VISIBLE_DEVICES` together for multi-GPU execution. The launcher uses `torch.distributed.run` and the code reads `LOCAL_RANK`.

Training outputs include `train_imagenet/outputs/stage*/example/checkpoint/model_best.pth.tar`. Evaluation loads the specified checkpoint and reports validation accuracy. The legacy shell entrypoints now delegate to these configurations and accept launcher overrides.

## Architecture search

```bash
# Offline: fit the primary accuracy predictor on supplied label arrays
python -m flashnas run --config configs/predictor-imagenet100.json

# Stage I: evolutionary search using that predictor
python -m flashnas run --config configs/search-stage1-imagenet100.json

# Inspect saved Pareto candidates
python -m flashnas run --config configs/plot-search.json

# Stage II: FPGA analytical latency-based selection
python -m flashnas run --config configs/search-stage2-fpga.json
```

Stage I saves `info.pth` and includes 20 search epochs, population 512, 128 crossover candidates, 128 mutation candidates, seed 2024, and an 80–90 million OP budget. The `--time_mode None` recipe uses OPs in Stage I. The `max-epochs` option controls evolutionary generations; it is unrelated to predictor-fitting epochs or final-network training epochs.

For local PyTorch timing of every candidate, use:

```bash
python -m flashnas run --config configs/search-stage2-pytorch.json
```

The timing recipe constructs every candidate's binary network and saves aligned architecture, accuracy, cost, and latency arrays in `info_second.pth`. Its default CPU settings use batch size 1, 20 warmup passes, 30 timed passes, and one compute thread, with latency reported in **milliseconds per batch**. CUDA timing synchronizes the selected device. See [measurement details](docs/reproducibility.md#measurement-scope) for experiment protocols.

## Reported results

ImageNet-100 results reported in the manuscript:

| Architecture | Target device | Top-1 (%) | Latency per image (ms, mean ± SD) |
| --- | --- | ---: | ---: |
| FlashNAS-BNN-G | CPU, Cortex-X4 | 81.40 | 164.57 ± 1.37 |
| FlashNAS-BNN-H | GPU, Mali-G77 | 81.86 | 48.45 ± 0.21 |
| FlashNAS-BNN-I | FPGA, XC7VX690T | 81.61 | 2.340 ± 0.008 |

The target-specific architectures combine competitive classification accuracy with efficient inference across CPU, GPU, and FPGA platforms.

## Project structure

```text
flashnas/                 Common CLI, run provenance, candidate-model construction
configs/                  Predictor, search, training, evaluation and plot recipes
net_predict/              CIF-Net and predictor baselines
net_bw/                   Bandwidth predictor used by CPU CLE
utils/                    Candidate encoding, cost models, datasets, logging and plots
train_imagenet/train/     Candidate networks and Apex training loop
script/                   Compatibility search entrypoint
train_imagenet/script/   Compatibility stage-1/stage-2 training entrypoints
data/                    Data contracts (large input files remain untracked)
checkpoints/             Checkpoint contracts (weights remain untracked)
outputs/                 Local run manifests and logs (untracked)
docs/                    Experiment protocols and contributor guidance
tests/                   CLI, artifact checks, dataset split and CPU model regressions
```

Numbered root scripts remain available for feature generation, architecture inspection, result aggregation, and maintenance. Use the configured entrypoints for repeatable training, search, evaluation, and plotting experiments.

## Reproducibility and development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

CPU tests verify candidate construction, disjoint deterministic predictor splits, input checks, configuration overrides, script paths, and Python syntax. GitHub Actions runs these checks on every push and pull request. [Contributor guidance](docs/contributing.md) explains how to add a configuration or backend without changing reported results.

The documentation layout follows [Papers with Code's release guidance](https://github.com/paperswithcode/releasing-research-code) and the [reproducible-paper Python template](https://github.com/klb2/reproducible-paper-python-template), adapted to this repository's actual artifacts.
