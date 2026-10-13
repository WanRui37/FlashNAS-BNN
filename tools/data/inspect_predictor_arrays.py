"""Inspect the shapes and dtypes of aligned predictor arrays."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    for name in ('2-np_best_accuracy.npy', '2-np_feature_single_vec.npy', '2-np_feature_total_vec.npy'):
        array = np.load(args.data_dir / name, mmap_mode='r', allow_pickle=False)
        print(f'{name}: shape={array.shape} dtype={array.dtype}')


if __name__ == '__main__':
    main()
