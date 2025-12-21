# FlashNAS-BNN

FlashNAS-BNN, a rapid chain-structured binary NAS framework built on three components.

## Environment dependency

- Python 3.10
- PyTorch 2.0.0+
- CUDA 11.7+
- NVIDIA Apex (版本要求: https://github.com/NVIDIA/apex/tree/24.04.01)

```bash
conda env create -f environment.yml -n torch_nas
conda activate torch_nas
```

### Install Apex

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 24.04.01
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```

### Search

```bash
bash script/example.sh
```

### Train

```bash
cd train_imagenet
bash script/stage1/run_bnn_imagenet100.sh
```