"""Build candidate-specific models from the original network implementation."""
import importlib
import os
from contextlib import contextmanager


@contextmanager
def architecture_environment(candidate):
    values = [int(x) for x in candidate]
    if not values or len(values) % 3:
        raise ValueError("Candidate must contain channels, strides and kernel types of equal length")
    count = len(values) // 3
    channels, strides, kernels = values[:count], values[count:2 * count], values[2 * count:]
    valid = [(c, s, k) for c, s, k in zip(channels, strides, kernels) if c != -1]
    if not valid or any(c <= 0 or s not in (1, 2) or k not in (0, 1) for c, s, k in valid):
        raise ValueError("Expected positive channels, strides 1/2 and kernels 0 (3x3)/1 (1x1)")
    # Constructors read ARCH_ALL at import time. Restore the caller's environment afterwards.
    old = os.environ.get("ARCH_ALL")
    os.environ["ARCH_ALL"] = str(tuple([x[0] for x in valid] + [x[1] for x in valid] + [x[2] for x in valid]))
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ARCH_ALL", None)
        else:
            os.environ["ARCH_ALL"] = old


def build_model(candidate, num_classes=100, stage=2):
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    module_name = f"train_imagenet.train.nets.reactnet_firstBConv_lbp_ther_48c_100_stage{stage}"
    with architecture_environment(candidate):
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        model = module.reactnet(num_classes=num_classes)
    return model
