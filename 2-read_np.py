import itertools
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch.cuda.amp import autocast as autocast

if __name__ == "__main__":
    best_accuracy_np = np.load("./2-np_best_accuracy.npy")
    feature_single_vec_np = np.load("./2-np_feature_single_vec.npy")
    feature_total_vec_np = np.load("./2-np_feature_total_vec.npy")
    print(f"best_accuracy_np.shape is {best_accuracy_np.shape}")
    print(f"feature_single_vec_np.shape is {feature_single_vec_np.shape}")
    print(f"feature_total_vec_np.shape is {feature_total_vec_np.shape}")
