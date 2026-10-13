# FlashNAS-BNN

**Chain-Structured Binary Neural Architecture Search for Efficient Visual Recognition across Platforms**

FlashNAS-BNN searches chain-structured binary neural networks using an accuracy predictor (CIF-Net), evolutionary candidate search, and hardware-aware candidate selection. This repository contains predictor training, two-stage search, and ImageNet-100 candidate training/evaluation code.

## Highlights

- Chain-structured BNN NAS for efficient visual recognition across platforms.
- A two-stage search with CIF-Net accelerates architecture search.
- Cross-platform latency estimation bridges the gap between OPs and real runtime.
- Image classification benchmarks show strong accuracy-latency trade-offs.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data and checkpoints](#data-and-checkpoints)
- [Training and evaluation](#training-and-evaluation)
- [Architecture search](#architecture-search)
- [Reported results](#reported-results)
- [Project structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Installation

The minimum requirements are Python 3.10, PyTorch 2.0.0, torchvision 0.15.0, and CUDA 11.7 for GPU execution. These are minimum version requirements, rather than the exact versions used for the paper's experiments; the experimental environment is described in the supplementary material. Use compatible PyTorch, torchvision, and CUDA builds. [environment.yml](environment.yml) provides a full environment snapshot, while [environment-minimal.yml](environment-minimal.yml) provides a compact setup. Run the following commands from the repository root:

```bash
git clone https://github.com/WanRui37/FlashNAS-BNN.git
cd FlashNAS-BNN
conda env create -f environment-minimal.yml
conda activate flashnas-bnn
python main.py doctor
```

Alternatively, create a Python 3.10 environment, install a matching PyTorch/torchvision build using the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/), then install the research dependencies:

```bash
python -m pip install -r requirements.txt
```

CPU execution is suitable for framework checks, predictor experiments, and local timing experiments. The existing image-training loop requires an NVIDIA GPU and NVIDIA Apex. Install Apex in a separate directory using the pinned revision:

```bash
git clone --branch 24.04.01 https://github.com/NVIDIA/apex.git /tmp/flashnas-apex
python -m pip install -v --disable-pip-version-check --no-build-isolation /tmp/flashnas-apex
```

GCN predictor baselines additionally use `python -m pip install -r requirements-graph.txt`. The primary `OUR` predictor uses the standard research dependencies. Run `doctor` to inspect the installed modules.

## Quick start

Run directly from the repository checkout; only the environment dependencies need to be installed. The project itself does not require installation.

Inspect an experiment without downloading data or starting GPU work:

```bash
python main.py run --config configs/examples/search-stage1-imagenet100.json --dry-run
bash run.sh run --config configs/examples/train-stage1-imagenet100.json --dry-run
```

All files in `configs/examples/` are illustrative examples for using the framework, rather than the official experiment configurations for the paper models. Copy an example and adapt its paths, architecture, budgets, and device settings for your experiment.

Every configuration is ordinary JSON. `--set` overrides an existing value; it rejects misspelled keys:

```bash
python main.py run --config configs/examples/search-stage1-imagenet100.json \
  --set parameters.seed=1234 \
  --set parameters.logdir=./search_output/imagenet100/seed1234/stage1 \
  --set run_dir=./outputs/runs/search-seed1234 --dry-run
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

**Pretrained model release:** Trained checkpoints for the FlashNAS-BNN models will be publicly released after the paper is accepted. Download links and model-specific loading instructions will be added here upon release.

## Training and evaluation

The example training recipes include an illustrative candidate architecture. Set `environment.ARCH_ALL` to train your selected architecture, using the **same** value for both stages. Architecture encoding is documented in [data/README.md](data/README.md).

```bash
# 1. Candidate training, stage 1: 128 epochs
python main.py run --config configs/examples/train-stage1-imagenet100.json

# 2. Candidate training, stage 2: initialize from stage-1 checkpoint
python main.py run --config configs/examples/train-stage2-imagenet100.json

# 3. Evaluate trained weights on the validation split
python main.py run --config configs/examples/evaluate-imagenet100.json
```

The examples use Adam, learning rate 1.25e-3, batch size 32 per process, seed 2024, and one GPU. Stage 1 uses weight decay 1e-5; stage 2 uses 0. Set `nproc` and `environment.CUDA_VISIBLE_DEVICES` together for multi-GPU execution. The launcher uses `torch.distributed.run` and the code reads `LOCAL_RANK`.

Training outputs include `train_imagenet/outputs/stage*/example/checkpoint/model_best.pth.tar`. Evaluation loads the specified checkpoint and reports validation accuracy. The legacy shell entrypoints now delegate to these configurations and accept launcher overrides.

## Architecture search

Generate an initial population for predictor training with seed 1234:

```bash
python main.py run --config configs/examples/init-imagenet100.json
```

The `Only-Init` recipe generates eight initial candidates and adds 50 maximal candidates without requiring a CIF-Net checkpoint. Boundary architectures deliberately broaden the fitting coverage beyond the later search budget. Architecture records are written to `init_output_path`; use new run and output paths for another initialization run.

```bash
# Offline: fit the primary accuracy predictor on supplied label arrays
python main.py run --config configs/examples/predictor-imagenet100.json

# Stage I: evolutionary search using that predictor
python main.py run --config configs/examples/search-stage1-imagenet100.json

# Inspect saved Pareto candidates
python main.py run --config configs/examples/plot-search.json

# Stage II: FPGA analytical latency-based selection
python main.py run --config configs/examples/search-stage2-fpga.json
```

Stage I saves `info.pth` and includes 20 search epochs, population 512, 128 crossover candidates, 128 mutation candidates, seed 2024, and an 80–90 million OP budget. The `--time_mode None` recipe uses OPs in Stage I. The `max-epochs` option controls evolutionary generations; it is unrelated to predictor-fitting epochs or final-network training epochs.

For local PyTorch timing of every candidate, use:

```bash
python main.py run --config configs/examples/search-stage2-pytorch.json
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
main.py                  Direct experiment entrypoint
run.sh                   Shell entrypoint from any working directory
flashnas/                 Common CLI, run provenance, candidate-model construction
flashnas/search/          Stage-one search, stage-two selection and Pareto plotting
tools/                    Data preparation, architecture analysis and maintenance tools
configs/examples/         Example configurations for the supported workflows
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

Search implementations live in `flashnas/search/`; supporting utilities are organized under `tools/data/`, `tools/analysis/`, and `tools/maintenance/` with descriptive `snake_case` names. The configured CLI remains the main entrypoint for training, search, evaluation, and plotting. See [tools and script migration](docs/tools.md) for the complete filename mapping and utility commands.

## Acknowledgements

We thank the following projects for sharing their research code and providing valuable references for this work:

- [NAS-BNN](https://github.com/VDIGPKU/NAS-BNN)
- [EBNAS](https://github.com/sscckk/EBNAS)
- [BNAS](https://github.com/gistvision/bnas)
