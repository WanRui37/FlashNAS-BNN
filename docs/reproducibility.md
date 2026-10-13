# Reproduction status

The release layout takes its structure from the research-code guides linked in the main README. The checks below describe actual availability, not a promise of exact numerical reproduction.

| Component | Available here | External input / remaining gap |
| --- | --- | --- |
| Environment | Compact Conda recipe, pip requirements and original export | Matching CUDA driver and Apex for image training |
| CIF-Net | Primary RNN plus optional predictor baselines | Measured architecture-label arrays and trained predictor weights |
| Stage I | Evolutionary search and explicit ImageNet-100 recipe | Matching CIF-Net checkpoint |
| Stage II / FPGA | Analytical model and candidate selection | Calibration, deployed FPGA implementation and bitstream |
| Stage II / CPU | Legacy bandwidth predictor and local PyTorch timing | Measured bandwidth checkpoint, paper deployment toolchain |
| Stage II / GPU | Local CUDA PyTorch timing can be configured | GPU CLE implementation and paper OpenCL deployment |
| Candidate training | ImageNet-100 network, two stages, Apex loop | Licensed images, exact paper class list and pretrained weights |
| Classification evaluation | Validation path that requires a checkpoint | Matching architecture/checkpoint and class mapping |
| Results | Author-reported manuscript table in CSV | Complete D–I model configs and releaseable trained weights |
| Tests | CPU mechanics and configuration checks | Full expensive training/search and hardware reproduction |

## Measurement scope

The revised manuscript reports batch size 32, input 224×224, 20 warmup runs and 50 timed runs, synchronization, excluding host preprocessing, and dividing batch latency by 32 to report per-image latency. The deployment setup uses CPU single-thread execution/affinity, Bolt v1.5.2/OpenCL for GPU, CPU/GPU maximum frequency settings of 3.3 GHz/836 MHz, and FPGA 180 MHz.

The provided local PyTorch timing example instead uses batch 1 and 30 timed iterations and reports milliseconds per batch. It does not implement the manuscript's ARM binary kernels, OpenCL kernels or FPGA design. It must be reported separately. Random initialization for architecture timing does not establish classification accuracy.

The predictor-labeling cost, online Stage-I/Stage-II search costs, and final network training costs are distinct. Example settings are taken from the checked-in scripts; final training budgets and benchmark groups must be matched to the particular paper comparison. Do not treat the 128+128 epoch ImageNet-100 example as a universal ImageNet-1000/CIFAR recipe.

## Behavior changes in this release

- Predictor train/validation indices now come from one permutation, preventing overlap. Existing paper predictor metrics were not recomputed.
- Prediction during search uses evaluation mode. Predictor checkpoint/output paths are now consistent.
- Local timing builds each candidate's actual network instead of importing a nonexistent shared model path. All candidates are retained for measurement.
- CPU-only predictor/search imports no longer require CUDA or the optional GCN backend.
- Legacy ONNX-log selection is rejected by the common launcher until candidate-ID alignment can be guaranteed; direct legacy scripts are preserved.

These fixes improve future experiments but do not assert that the changed code reproduces old numerical outcomes unchanged.

## Code provenance

`net_bw/Predict_BW.py`, `net_bw/dataset_bw.py` and `net_bw/__init__.py` were restored from the user's companion repository [WanRui37/FlashNAS-BNN-myself](https://github.com/WanRui37/FlashNAS-BNN-myself/tree/d4ac3d0cf20589324bbb1a557cfacd8534edd050/net_bw), commit `d4ac3d0cf20589324bbb1a557cfacd8534edd050`, because the original public scripts imported these missing modules. No bandwidth data or trained weights were invented or copied as new measurements.

The README results table preserves the author-provided revised manuscript table (`tab:4`) from the associated paper workspace at parent commit `c9dab26`. It is a report of existing results, not output of the tests. Publication metadata and license require confirmation before an archival release.
