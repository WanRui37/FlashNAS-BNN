import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch.cuda.amp import autocast as autocast

torch.manual_seed(2024)


class Basic_Block(nn.Module):
    def __init__(self, embed_dim_1, embed_dim_2):
        super(Basic_Block, self).__init__()
        self.Linear = nn.Linear(embed_dim_1, embed_dim_2)
        self.BN = nn.BatchNorm1d(embed_dim_2)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        out = self.Linear(x)
        out = self.BN(out)
        out = self.PReLU(out)
        return out


class Basic_Block_no_bn(nn.Module):
    def __init__(self, embed_dim_1, embed_dim_2):
        super(Basic_Block_no_bn, self).__init__()
        self.Linear = nn.Linear(embed_dim_1, embed_dim_2)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        out = self.Linear(x)
        out = self.PReLU(out)
        return out


class Predict_BW(nn.Module):
    def __init__(self, embed_dim_1, embed_dim_2, hidden_size=256):
        super(Predict_BW, self).__init__()
        self.embed_dim_1 = embed_dim_1
        self.embed_dim_2 = embed_dim_2
        self.hidden_size = hidden_size

        self.linear1 = nn.Sequential(
            Basic_Block(self.embed_dim_1, self.hidden_size),
            Basic_Block(self.hidden_size, self.hidden_size),
            Basic_Block(self.hidden_size, self.hidden_size),
        )

        self.linear2 = nn.Sequential(
            Basic_Block_no_bn(self.embed_dim_2, self.hidden_size),
            Basic_Block_no_bn(self.hidden_size, self.hidden_size),
            Basic_Block_no_bn(self.hidden_size, self.hidden_size),
        )

        # self.linear3 = nn.Sequential(
        #     Basic_Block_no_bn(self.embed_dim_2//2, self.hidden_size),
        #     Basic_Block_no_bn(self.hidden_size, self.hidden_size),
        # )

        self.linear4 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x):

        out_final_1 = self.linear1(x[:, : self.embed_dim_1])
        out_final_2 = self.linear2(x[:, self.embed_dim_1 :])
        # out_final_3 = self.linear3(x[:,self.embed_dim_1+self.embed_dim_2//2:])

        out_final = torch.concat((out_final_1, out_final_2), dim=1)

        # out_final = out_final_1 * out_final_2
        # out_final = out_final_1 + out_final_2
        # out_final = torch.concat((out_final_1,out_final_2), dim=1)

        out_final = self.linear4(out_final)
        out_final = out_final.view(-1)
        # print(out_final.shape)
        return out_final
