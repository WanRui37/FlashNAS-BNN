import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config


def normalization_simple(feature, mean, std):
    # print(f"feature_all.shape:{feature.shape}!!!")
    unlogged_part = torch.log10(feature[:, 0:4])
    logged_part = feature[:, 4:]
    all_part = torch.cat((unlogged_part, logged_part), dim=-1)
    all_part = (all_part - mean) / std
    return all_part


def normalization_complex(feature, mean, std):
    # print(f"feature_single.shape:{feature.shape}!!!")
    unlogged_part = feature[:, :, 0:4]
    logged_part = torch.log10(feature[:, :, 4:8])
    all_part = torch.cat((unlogged_part, logged_part), dim=-1)
    all_part = (all_part - mean) / std
    return all_part


class TextDataset:
    def __init__(self, best_accuracy_torch, feature_single_vec_torch, feature_total_vec_torch, layer_len_max):
        self.best_accuracy_torch = best_accuracy_torch.reshape(-1, 1)
        feature_single_vec_torch = feature_single_vec_torch + 1e-5
        self.feature_single_vec_torch = feature_single_vec_torch[:, 0:layer_len_max, :]
        self.feature_total_vec_torch = feature_total_vec_torch
        """
                [channel_[ii]] + conv_one_hot + [flops_single[ii]] + \
                [bops_single[ii]] + [flparm_single[ii]] + [bparm_single[ii]]
                需要对这feature_single_vec_torch和feature_total_vec_torch做一个归一化

                feature_total_vec = [flops_total] + [bops_total] + [flparm_total]
                + [bparm_total] + [len(stage_out_channels)] + unique_channels
        """

        # print(f"best_accuracy_torch.shape:{self.best_accuracy_torch.shape}")

        batch_size, layer_max, feature_num = feature_single_vec_torch.shape
        # print(f"feature_single_vec_torch.shape:{feature_single_vec_torch.shape}")
        # print(f"feature_single_vec_1.mean:{torch.mean(feature_single_vec_torch[:,:,0:4].view(batch_size*layer_max, -1), dim=0)}")
        # print(f"feature_single_vec_2.mean:{torch.mean(torch.log10(feature_single_vec_torch[:,:,4:8].view(batch_size*layer_max, -1)), dim=0)}")
        # print(f"feature_single_vec_1.std:{torch.std(feature_single_vec_torch[:,:,0:4].view(batch_size*layer_max, -1), dim=0)}")
        # print(f"feature_single_vec_2.std:{torch.std(torch.log10(feature_single_vec_torch[:,:,4:8].view(batch_size*layer_max, -1)), dim=0)}")

        # print(f"feature_total_vec_torch.shape:{self.feature_total_vec_torch.shape}")
        # print(f"feature_total_vec_1.mean:{torch.mean(torch.log10(feature_total_vec_torch), dim=0)}")
        # print(f"feature_total_vec_1.std:{torch.std(torch.log10(feature_total_vec_torch), dim=0)}")

        mean11 = torch.FloatTensor([0, 0, 0, 0])
        mean12 = torch.FloatTensor([2.5697, 2.3792, 0.5134, 0.5379])
        # mean11 = torch.FloatTensor([0, 0, 0, 0])
        # mean12 = torch.FloatTensor([0, 0, 0, 0])
        mean1 = torch.cat([mean11, mean12], dim=0)

        std11 = torch.FloatTensor([1, 1, 1, 1])
        std12 = torch.FloatTensor([5.2759, 5.5251, 3.8568, 4.1432])
        # std11 = torch.FloatTensor([1, 1, 1, 1])
        # std12 = torch.FloatTensor([1, 1, 1, 1])
        std1 = torch.cat([std11, std12], dim=0)

        self.feature_single_vec_torch = normalization_complex(self.feature_single_vec_torch, mean1, std1)

        # mean2 = torch.FloatTensor([0]*11)
        # std2 = torch.FloatTensor([1]*11)
        mean2 = torch.FloatTensor([7.7408, 8.0334, 4.7231, 5.0934] + [0] * 7)
        std2 = torch.FloatTensor([0.1823, 0.2967, 0.2232, 0.2088] + [1] * 7)
        self.feature_total_vec_torch = normalization_simple(self.feature_total_vec_torch, mean2, std2)
        # exit(0)

        _, _, single_vec = self.feature_single_vec_torch.shape
        total_len, total_vec = self.feature_total_vec_torch.shape
        print(f"单个特征的大小为:{self.feature_single_vec_torch.shape}")
        print(f"总体特征的大小为:{self.feature_total_vec_torch.shape}")

        if single_vec > total_vec:
            padding_size = single_vec - total_vec
            padding = torch.zeros(total_len, padding_size)
            self.feature_total_vec_torch = torch.cat((self.feature_total_vec_torch, padding), dim=1)
        elif single_vec < total_vec:
            padding_size = total_vec - single_vec
            padding = torch.zeros(total_len, layer_len_max, padding_size)
            self.feature_single_vec_torch = torch.cat((self.feature_single_vec_torch, padding), dim=2)

        self.x = torch.concat((self.feature_single_vec_torch, self.feature_total_vec_torch.unsqueeze(1)), dim=1)

    def __getitem__(self, index):
        x = self.x[index, :, :]
        y2 = self.best_accuracy_torch[index, :]
        y = y2

        return x, y

    def __len__(self):
        return self.best_accuracy_torch.shape[0]


class TextDataset_cifar:
    def __init__(self, best_accuracy_torch, feature_single_vec_torch, feature_total_vec_torch, layer_len_max):
        self.best_accuracy_torch = best_accuracy_torch.reshape(-1, 1)
        feature_single_vec_torch = feature_single_vec_torch + 1e-5
        self.feature_single_vec_torch = feature_single_vec_torch[:, 0:layer_len_max, :]
        self.feature_total_vec_torch = feature_total_vec_torch
        """
                [channel_[ii]] + conv_one_hot + [flops_single[ii]] + \
                [bops_single[ii]] + [flparm_single[ii]] + [bparm_single[ii]]
                需要对这feature_single_vec_torch和feature_total_vec_torch做一个归一化

                feature_total_vec = [flops_total] + [bops_total] + [flparm_total]
                + [bparm_total] + [len(stage_out_channels)] + unique_channels
        """

        # print(f"best_accuracy_torch.shape:{self.best_accuracy_torch.shape}")

        batch_size, layer_max, feature_num = feature_single_vec_torch.shape
        # print(f"feature_single_vec_torch.shape:{feature_single_vec_torch.shape}")
        # print(f"feature_single_vec_1.mean:{torch.mean(feature_single_vec_torch[:,:,0:4].view(batch_size*layer_max, -1), dim=0)}")
        # print(f"feature_single_vec_2.mean:{torch.mean(torch.log10(feature_single_vec_torch[:,:,4:8].view(batch_size*layer_max, -1)), dim=0)}")
        # print(f"feature_single_vec_1.std:{torch.std(feature_single_vec_torch[:,:,0:4].view(batch_size*layer_max, -1), dim=0)}")
        # print(f"feature_single_vec_2.std:{torch.std(torch.log10(feature_single_vec_torch[:,:,4:8].view(batch_size*layer_max, -1)), dim=0)}")

        # print(f"feature_total_vec_torch.shape:{self.feature_total_vec_torch.shape}")
        # print(f"feature_total_vec_1.mean:{torch.mean(torch.log10(feature_total_vec_torch), dim=0)}")
        # print(f"feature_total_vec_1.std:{torch.std(torch.log10(feature_total_vec_torch), dim=0)}")

        mean11 = torch.FloatTensor([0, 0, 0, 0])
        mean12 = torch.FloatTensor([2.7289, 2.5743, 1.0837, 1.1290])
        # mean11 = torch.FloatTensor([0, 0, 0, 0])
        # mean12 = torch.FloatTensor([0, 0, 0, 0])
        mean1 = torch.cat([mean11, mean12], dim=0)

        std11 = torch.FloatTensor([1, 1, 1, 1])
        std12 = torch.FloatTensor([4.5983, 4.9710, 3.6195, 4.0246])
        # std11 = torch.FloatTensor([1, 1, 1, 1])
        # std12 = torch.FloatTensor([1, 1, 1, 1])
        std1 = torch.cat([std11, std12], dim=0)

        self.feature_single_vec_torch = normalization_complex(self.feature_single_vec_torch, mean1, std1)

        # mean2 = torch.FloatTensor([0]*11)
        # std2 = torch.FloatTensor([1]*11)
        mean2 = torch.FloatTensor([6.9168, 7.2103, 4.5952, 5.1106] + [0] * 5)
        std2 = torch.FloatTensor([0.0968, 0.0990, 0.1245, 0.1377] + [1] * 5)
        self.feature_total_vec_torch = normalization_simple(self.feature_total_vec_torch, mean2, std2)
        # exit(0)

        _, _, single_vec = self.feature_single_vec_torch.shape
        total_len, total_vec = self.feature_total_vec_torch.shape
        print(f"单个特征的大小为:{self.feature_single_vec_torch.shape}")
        print(f"总体特征的大小为:{self.feature_total_vec_torch.shape}")

        if single_vec > total_vec:
            padding_size = single_vec - total_vec
            padding = torch.zeros(total_len, padding_size)
            self.feature_total_vec_torch = torch.cat((self.feature_total_vec_torch, padding), dim=1)
        elif single_vec < total_vec:
            padding_size = total_vec - single_vec
            padding = torch.zeros(total_len, layer_len_max, padding_size)
            self.feature_single_vec_torch = torch.cat((self.feature_single_vec_torch, padding), dim=2)

        self.x = torch.concat((self.feature_single_vec_torch, self.feature_total_vec_torch.unsqueeze(1)), dim=1)

    def __getitem__(self, index):
        x = self.x[index, :, :]
        y2 = self.best_accuracy_torch[index, :]
        y = y2

        return x, y

    def __len__(self):
        return self.best_accuracy_torch.shape[0]


def get_train_set(np_best_accuracy_path, np_feature_single_vec_path, np_feature_total_vec_path, dataset):

    best_accuracy_np = np.load(np_best_accuracy_path)
    feature_single_vec_np = np.load(np_feature_single_vec_path)
    feature_total_vec_np = np.load(np_feature_total_vec_path)

    best_accuracy_torch = torch.from_numpy(best_accuracy_np)
    feature_single_vec_torch = torch.from_numpy(feature_single_vec_np)
    feature_total_vec_torch = torch.from_numpy(feature_total_vec_np)

    train_set = TextDataset(best_accuracy_torch, feature_single_vec_torch, feature_total_vec_torch)

    return train_set


def get_train_valid_set(np_best_accuracy_path, np_feature_single_vec_path, np_feature_total_vec_path, dataset, ratio):

    best_accuracy_np = np.load(np_best_accuracy_path)
    feature_single_vec_np = np.load(np_feature_single_vec_path)
    feature_total_vec_np = np.load(np_feature_total_vec_path)

    best_accuracy_torch = torch.from_numpy(best_accuracy_np)
    feature_single_vec_torch = torch.from_numpy(feature_single_vec_np)
    feature_total_vec_torch = torch.from_numpy(feature_total_vec_np)

    print(f"best_accuracy_torch.shape:{best_accuracy_torch.shape}")
    print(f"feature_single_vec_torch.shape:{feature_single_vec_torch.shape}")
    print(f"feature_total_vec_torch.shape:{feature_total_vec_torch.shape}")

    if not 0 < ratio < 1:
        raise ValueError("ratio must be strictly between 0 and 1")
    sample_count = best_accuracy_torch.shape[0]
    if feature_single_vec_torch.shape[0] != sample_count or feature_total_vec_torch.shape[0] != sample_count:
        raise ValueError("Predictor arrays must contain the same number of samples")
    split = int(sample_count * ratio)
    if split == 0 or split == sample_count:
        raise ValueError("Predictor split requires nonempty training and validation sets")
    indices = torch.randperm(sample_count)
    train_idx = indices[:split]
    valid_idx = indices[split:]
    print(f"train_idx.shape:{train_idx.shape}")
    print(f"valid_idx.shape:{valid_idx.shape}")
    # exit()

    if dataset == "imagenet100":
        layer_len_max = 32
        train_set = TextDataset(
            best_accuracy_torch[train_idx],
            feature_single_vec_torch[train_idx, :, :],
            feature_total_vec_torch[train_idx, :],
            layer_len_max,
        )

        valid_set = TextDataset(
            best_accuracy_torch[valid_idx],
            feature_single_vec_torch[valid_idx, :, :],
            feature_total_vec_torch[valid_idx, :],
            layer_len_max,
        )
    elif dataset in ["cifar10", "cifar100"]:
        layer_len_max = 25
        train_set = TextDataset_cifar(
            best_accuracy_torch[train_idx],
            feature_single_vec_torch[train_idx, :, :],
            feature_total_vec_torch[train_idx, :],
            layer_len_max,
        )

        valid_set = TextDataset_cifar(
            best_accuracy_torch[valid_idx],
            feature_single_vec_torch[valid_idx, :, :],
            feature_total_vec_torch[valid_idx, :],
            layer_len_max,
        )

    return train_set, valid_set
