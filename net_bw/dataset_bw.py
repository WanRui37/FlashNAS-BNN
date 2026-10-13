import numpy as np
import torch
from torch.utils.data import DataLoader


def get_one_hot(row_dict_1):
    tensor_kernel = torch.tensor(row_dict_1, dtype=torch.float32)
    unique_values = torch.unique(tensor_kernel)
    one_hot_classes = unique_values.shape[0]
    print(f"one_hot_classes:{one_hot_classes}")
    batch_size = tensor_kernel.numel()

    one_hot_result = torch.zeros(batch_size, one_hot_classes, dtype=torch.float32).cuda()
    for index, element in enumerate(tensor_kernel.view(-1)):
        # 找到元素在唯一值中的位置索引
        class_index = (unique_values == element).nonzero(as_tuple=True)[0]
        # print(class_index)
        one_hot_result[index, class_index] = 1.0
    return one_hot_result, one_hot_classes


def normalization_complex(feature, mean, std, embed_dim_1):
    unlogged_part = feature[:, 0:embed_dim_1] + 1e-5
    logged_part = feature[:, embed_dim_1:]

    all_part = torch.cat((unlogged_part, logged_part), dim=-1)
    all_part = (all_part - mean) / std

    return all_part


def normalization_simple(feature, embed_dim_1):
    unlogged_part = torch.log2(feature[:, 0:embed_dim_1])
    logged_part = feature[:, embed_dim_1:]
    all_part = torch.cat((unlogged_part, logged_part), dim=-1)

    return all_part


class BWDataset:
    def __init__(self, x, y, embed_dim_1):
        self.x = x

        self.x = normalization_simple(x, embed_dim_1)

        # batch_size, feature_num = self.x.shape
        # print(f"x.shape:{self.x.shape}")
        # print(f"x.mean:{torch.mean(self.x[:,0:embed_dim_1].view(batch_size, -1), dim=0)}")
        # print(f"x.std:{torch.std(self.x[:,0:embed_dim_1].view(batch_size, -1), dim=0)}")

        # mean11 = torch.FloatTensor([4.0688e+01, 3.4109e+02, 3.3906e+01, 3.4200e+02, 1.5324e+08, 8.8294e+05,
        #                             1.8502e+05, 1.5758e+05]).to(self.x.device)
        # mean12 = torch.FloatTensor([0]*2).to(self.x.device)
        # mean1 = torch.cat([mean11, mean12], dim=0)

        # std11 = torch.FloatTensor([4.7424e+01, 2.6682e+02, 3.3974e+01, 2.6568e+02, 1.8460e+08, 1.7947e+06,
        #                             1.6869e+05, 1.2938e+05]).to(self.x.device)
        # std12 = torch.FloatTensor([1]*2).to(self.x.device)
        # std1 = torch.cat([std11, std12], dim=0)

        # self.x = normalization_complex(self.x, mean1, std1, embed_dim_1)

        self.y = torch.log2(y)
        # self.y = y

    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index, :]

        return x, y

    def __len__(self):
        return self.x.shape[0]
