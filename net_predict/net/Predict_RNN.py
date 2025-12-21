import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch_geometric.nn import GCNConv, GINConv, GINEConv, Sequential

torch.manual_seed(2024)
torch.backends.cudnn.benchmark = True


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableScale(nn.Module):
    def __init__(self, out_chn):
        super(LearnableScale, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, out_chn), requires_grad=True)

    def forward(self, x):
        out = x * self.scale.expand_as(x)
        return out


class LearnableAttention(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(LearnableAttention, self).__init__()
        self.att = nn.Linear(in_chn, out_chn)
        self.relu = nn.PReLU()

    def forward(self, x, h=None):
        out = self.att(x)
        out = self.relu(out)
        if h is not None:
            out = out + h
        return out


class Predict_GCN(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device):
        super(Predict_GCN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.feature_single_num = 8
        # self.feature_single_num = 4
        # self.feature_single_num = 6
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]

        for i in range(1, 7):
            self.add_module(
                f"gcn_conv{i}",
                nn.ModuleList(
                    [GCNConv(self.feature_single_num, self.hidden_size), nn.BatchNorm1d(self.hidden_size), nn.PReLU()]
                ),
            )

        self.embedding1 = nn.Sequential(
            nn.Linear(self.hidden_size * 6, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size * 2, 1)
        )

    def create_edge_index(self, num):
        first_row = torch.arange(0, num).to(self.device)
        second_row = torch.arange(1, num).to(self.device)
        second_row = torch.cat((second_row, torch.tensor([num - 1]).to(self.device)))
        return torch.stack((first_row, second_row))

    def forward(self, x):
        batch = x.shape[0]
        y = x[:, -1, :]
        stage1_num = int(y[:, 5].item())
        stage2_num = int(y[:, 6].item())
        stage3_num = int(y[:, 7].item())
        stage4_num = int(y[:, 8].item())
        stage5_num = int(y[:, 9].item())
        stage6_num = int(y[:, 10].item())
        total_num = stage1_num + stage2_num + stage3_num + stage4_num + stage5_num + stage6_num

        x = x[:, :, : self.feature_single_num]

        x1 = x[:, :stage1_num, :].view(-1, self.feature_single_num)
        x2 = x[:, stage1_num : stage1_num + stage2_num, :].view(-1, self.feature_single_num)
        x3 = x[:, stage1_num + stage2_num : stage1_num + stage2_num + stage3_num, :].view(-1, self.feature_single_num)
        x4 = x[:, stage1_num + stage2_num + stage3_num : stage1_num + stage2_num + stage3_num + stage4_num, :].view(
            -1, self.feature_single_num
        )
        x5 = x[
            :,
            stage1_num
            + stage2_num
            + stage3_num
            + stage4_num : stage1_num
            + stage2_num
            + stage3_num
            + stage4_num
            + stage5_num,
            :,
        ].view(-1, self.feature_single_num)
        x6 = x[:, stage1_num + stage2_num + stage3_num + stage4_num + stage5_num :, :].view(
            -1, self.feature_single_num
        )

        edge_index1 = self.create_edge_index(stage1_num)
        edge_index2 = self.create_edge_index(stage2_num)
        edge_index3 = self.create_edge_index(stage3_num)
        edge_index4 = self.create_edge_index(stage4_num)
        edge_index5 = self.create_edge_index(stage5_num)
        edge_index6 = self.create_edge_index(stage6_num)

        if stage1_num == 1:
            x1 = torch.zeros((stage1_num, self.hidden_size), device=x1.device)
        else:
            for i, run_single in enumerate(self.gcn_conv1):
                if i == 0:
                    x1 = run_single(x1, edge_index1)
                else:
                    x1 = run_single(x1)

        if stage2_num == 1:
            x2 = torch.zeros((stage2_num, self.hidden_size), device=x2.device)
        else:
            for i, run_single in enumerate(self.gcn_conv2):
                if i == 0:
                    x2 = run_single(x2, edge_index2)
                else:
                    x2 = run_single(x2)

        if stage3_num == 1:
            x3 = torch.zeros((stage3_num, self.hidden_size), device=x3.device)
        else:
            for i, run_single in enumerate(self.gcn_conv3):
                if i == 0:
                    x3 = run_single(x3, edge_index3)
                else:
                    x3 = run_single(x3)

        if stage4_num == 1:
            x4 = torch.zeros((stage4_num, self.hidden_size), device=x4.device)
        else:
            for i, run_single in enumerate(self.gcn_conv4):
                if i == 0:
                    x4 = run_single(x4, edge_index4)
                else:
                    x4 = run_single(x4)

        if stage5_num == 1:
            x5 = torch.zeros((stage5_num, self.hidden_size), device=x5.device)
        else:
            for i, run_single in enumerate(self.gcn_conv5):
                if i == 0:
                    x5 = run_single(x5, edge_index5)
                else:
                    x5 = run_single(x5)

        if stage6_num == 1:
            x6 = torch.zeros((stage6_num, self.hidden_size), device=x6.device)
        else:
            for i, run_single in enumerate(self.gcn_conv6):
                if i == 0:
                    x6 = run_single(x6, edge_index6)
                else:
                    x6 = run_single(x6)

        out_final = torch.cat(
            (
                x1[-1, :].view(1, -1),
                x2[-1, :].view(1, -1),
                x3[-1, :].view(1, -1),
                x4[-1, :].view(1, -1),
                x5[-1, :].view(1, -1),
                x6[-1, :].view(1, -1),
            ),
            dim=1,
        )
        out_final = self.embedding1(out_final)

        out2 = self.embedding2(y[:, self.feature_total_num_array])

        out_final = torch.cat([out_final, out2], dim=1)

        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final


class Predict_GCN_cifar(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device):
        super(Predict_GCN_cifar, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.feature_single_num = 8
        # self.feature_single_num = 4
        # self.feature_single_num = 6
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]

        for i in range(1, 5):
            self.add_module(
                f"gcn_conv{i}",
                nn.ModuleList(
                    [GCNConv(self.feature_single_num, self.hidden_size), nn.BatchNorm1d(self.hidden_size), nn.PReLU()]
                ),
            )

        self.embedding1 = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size * 2, 1)
        )

    def create_edge_index(self, num):
        first_row = torch.arange(0, num).to(self.device)
        second_row = torch.arange(1, num).to(self.device)
        second_row = torch.cat((second_row, torch.tensor([num - 1]).to(self.device)))
        return torch.stack((first_row, second_row))

    def forward(self, x):
        batch = x.shape[0]
        y = x[:, -1, :]
        stage1_num = int(y[:, 5].item())
        stage2_num = int(y[:, 6].item())
        stage3_num = int(y[:, 7].item())
        stage4_num = int(y[:, 8].item())
        total_num = stage1_num + stage2_num + stage3_num + stage4_num

        x = x[:, :, : self.feature_single_num]

        x1 = x[:, :stage1_num, :].view(-1, self.feature_single_num)
        x2 = x[:, stage1_num : stage1_num + stage2_num, :].view(-1, self.feature_single_num)
        x3 = x[:, stage1_num + stage2_num : stage1_num + stage2_num + stage3_num, :].view(-1, self.feature_single_num)
        x4 = x[:, stage1_num + stage2_num + stage3_num : stage1_num + stage2_num + stage3_num + stage4_num, :].view(
            -1, self.feature_single_num
        )

        edge_index1 = self.create_edge_index(stage1_num)
        edge_index2 = self.create_edge_index(stage2_num)
        edge_index3 = self.create_edge_index(stage3_num)
        edge_index4 = self.create_edge_index(stage4_num)

        if stage1_num == 1:
            x1 = torch.zeros((stage1_num, self.hidden_size), device=x1.device)
        else:
            for i, run_single in enumerate(self.gcn_conv1):
                if i == 0:
                    x1 = run_single(x1, edge_index1)
                else:
                    x1 = run_single(x1)

        if stage2_num == 1:
            x2 = torch.zeros((stage2_num, self.hidden_size), device=x2.device)
        else:
            for i, run_single in enumerate(self.gcn_conv2):
                if i == 0:
                    x2 = run_single(x2, edge_index2)
                else:
                    x2 = run_single(x2)

        if stage3_num == 1:
            x3 = torch.zeros((stage3_num, self.hidden_size), device=x3.device)
        else:
            for i, run_single in enumerate(self.gcn_conv3):
                if i == 0:
                    x3 = run_single(x3, edge_index3)
                else:
                    x3 = run_single(x3)

        if stage4_num == 1:
            x4 = torch.zeros((stage4_num, self.hidden_size), device=x4.device)
        else:
            for i, run_single in enumerate(self.gcn_conv4):
                if i == 0:
                    x4 = run_single(x4, edge_index4)
                else:
                    x4 = run_single(x4)

        out_final = torch.cat(
            (x1[-1, :].view(1, -1), x2[-1, :].view(1, -1), x3[-1, :].view(1, -1), x4[-1, :].view(1, -1)), dim=1
        )
        out_final = self.embedding1(out_final)

        out2 = self.embedding2(y[:, self.feature_total_num_array])

        out_final = torch.cat([out_final, out2], dim=1)

        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final


class Predict_peephole(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device):
        super(Predict_peephole, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.feature_single_num = 8
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]

        self.embedding = nn.Linear(self.feature_single_num, embed_dim)

        self.rnn1 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn2 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn3 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn4 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )
        self.rnn5 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )
        self.rnn6 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )

        self.embedding1 = nn.Sequential(
            nn.Linear(self.hidden_size * 6, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        batch = x.shape[0]
        y = x[:, -1, :]
        stage1_num = y[:, 5]
        stage2_num = y[:, 6]
        stage3_num = y[:, 7]
        stage4_num = y[:, 8]
        stage5_num = y[:, 9]
        stage6_num = y[:, 10]
        # print(f"x.shape:{x.shape}")

        # feature_embed = self.embedding(x)  # (batch, len, embed) (2, 28, 8)
        # print(f"feature_embed.shape:{feature_embed.shape}")
        feature_embed = x[:, :, : self.feature_single_num]

        layer_index_all = 0
        layer_index1 = 0
        for run_single in self.rnn1:
            if layer_index1 == 0:
                out1, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index1 < stage1_num:
                out1, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index1 = layer_index1 + 1
            layer_index_all = layer_index_all + 1

        layer_index2 = 0
        for run_single in self.rnn2:
            if layer_index2 == 0:
                out2, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index2 < stage2_num:
                out2, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index2 = layer_index2 + 1
            layer_index_all = layer_index_all + 1

        layer_index3 = 0
        for run_single in self.rnn3:
            if layer_index3 == 0:
                out3, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index3 < stage3_num:
                out3, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index3 = layer_index3 + 1
            layer_index_all = layer_index_all + 1

        layer_index4 = 0
        for run_single in self.rnn4:
            if layer_index4 == 0:
                out4, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index4 < stage4_num:
                out4, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index4 = layer_index4 + 1
            layer_index_all = layer_index_all + 1

        layer_index5 = 0
        for run_single in self.rnn5:
            if layer_index5 == 0:
                out5, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index5 < stage5_num:
                out5, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index5 = layer_index5 + 1
            layer_index_all = layer_index_all + 1

        layer_index6 = 0
        for run_single in self.rnn6:
            if layer_index6 == 0:
                out6, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index6 < stage6_num:
                out6, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index6 = layer_index6 + 1
            layer_index_all = layer_index_all + 1

        # print(f"out1.shape:{out1.shape}")
        # exit()

        # print(f"out1.shape:{out1.shape}")
        out_final = torch.cat((out1, out2, out3, out4, out5, out6), dim=1)
        # print(f"out_final.shape:{out_final.shape}")
        out_final = self.embedding1(out_final)

        # # (batch, 5)
        # # print(f"y[:, self.feature_total_num_array].shape:{y[:, self.feature_total_num_array].shape}")
        # out2 = self.embedding2(y[:, self.feature_total_num_array])
        # # (batch, hid)
        # out_final = torch.cat([out_final, out2], dim=1)
        # # (batch, hid*2)

        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final


class Predict_peephole_cifar(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device):
        super(Predict_peephole_cifar, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.feature_single_num = 8
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]

        self.embedding = nn.Linear(self.feature_single_num, embed_dim)

        self.rnn1 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn2 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn3 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn4 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )

        self.embedding1 = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        batch = x.shape[0]
        y = x[:, -1, :]
        stage1_num = y[:, 5]
        stage2_num = y[:, 6]
        stage3_num = y[:, 7]
        stage4_num = y[:, 8]
        # print(f"x.shape:{x.shape}")

        # feature_embed = self.embedding(x)  # (batch, len, embed) (2, 28, 8)
        # print(f"feature_embed.shape:{feature_embed.shape}")
        feature_embed = x[:, :, : self.feature_single_num]

        layer_index_all = 0
        layer_index1 = 0
        for run_single in self.rnn1:
            if layer_index1 == 0:
                out1, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index1 < stage1_num:
                out1, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index1 = layer_index1 + 1
            layer_index_all = layer_index_all + 1

        layer_index2 = 0
        for run_single in self.rnn2:
            if layer_index2 == 0:
                out2, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index2 < stage2_num:
                out2, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index2 = layer_index2 + 1
            layer_index_all = layer_index_all + 1

        layer_index3 = 0
        for run_single in self.rnn3:
            if layer_index3 == 0:
                out3, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index3 < stage3_num:
                out3, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index3 = layer_index3 + 1
            layer_index_all = layer_index_all + 1

        layer_index4 = 0
        for run_single in self.rnn4:
            if layer_index4 == 0:
                out4, (hn, cn) = run_single(feature_embed[:, layer_index_all, :])  # (batch, len, hidden)
            elif layer_index4 < stage4_num:
                out4, (hn, cn) = run_single(feature_embed[:, layer_index_all, :], (hn, cn))
            else:
                break
            layer_index4 = layer_index4 + 1
            layer_index_all = layer_index_all + 1

        # print(f"out1.shape:{out1.shape}")
        # exit()

        # print(f"out1.shape:{out1.shape}")
        out_final = torch.cat((out1, out2, out3, out4), dim=1)
        # print(f"out_final.shape:{out_final.shape}")
        out_final = self.embedding1(out_final)

        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final


class Predict_harsbnn(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device, layer_len_max=32):
        super(Predict_harsbnn, self).__init__()
        self.hidden_size = 128
        self.device = device
        self.feature_single_num = 8
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]
        self.embedding = nn.Linear(self.feature_single_num, embed_dim)
        self.rnn1 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn2 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn3 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(10 + 1)
            ]
        )
        self.rnn4 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )
        self.rnn5 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )
        self.rnn6 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(20 + 1)
            ]
        )
        self.embedding1 = nn.Sequential(
            nn.Linear(self.hidden_size * 6, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size * 2, 1)
        )

    def rnn_worker(self, rnn_list, feature_embed, stage_num):
        layer_index = 0
        for run_single in rnn_list:
            if layer_index == 0:
                out, (hn, cn) = run_single(feature_embed[:, layer_index, :])  # (batch, len, hidden)
            elif layer_index < stage_num:
                out, (hn, cn) = run_single(feature_embed[:, layer_index, :], (hn, cn))
            else:
                break
            layer_index = layer_index + 1
        return out

    def forward(self, x):
        x = x.to(self.device)

        batch, len, vec = x.shape
        y = x[:, -1, :]
        stage1_num = int(y[:, 5].item())
        stage2_num = int(y[:, 6].item())
        stage3_num = int(y[:, 7].item())
        stage4_num = int(y[:, 8].item())
        stage5_num = int(y[:, 9].item())
        stage6_num = int(y[:, 10].item())
        # print(f"stage1_num:{stage1_num}")
        # print(f"stage2_num:{stage2_num}")
        # print(f"stage3_num:{stage3_num}")
        # print(f"stage4_num:{stage4_num}")
        # print(f"stage5_num:{stage5_num}")
        # print(f"stage6_num:{stage6_num}")

        x = x[:, :, : self.feature_single_num]
        x1 = x[:, :stage1_num, :].reshape(-1, stage1_num, self.feature_single_num)
        x2 = x[:, stage1_num : stage1_num + stage2_num, :].reshape(-1, stage2_num, self.feature_single_num)
        x3 = x[:, stage1_num + stage2_num : stage1_num + stage2_num + stage3_num, :].reshape(
            -1, stage3_num, self.feature_single_num
        )
        x4 = x[:, stage1_num + stage2_num + stage3_num : stage1_num + stage2_num + stage3_num + stage4_num, :].reshape(
            -1, stage4_num, self.feature_single_num
        )
        x5 = x[
            :,
            stage1_num
            + stage2_num
            + stage3_num
            + stage4_num : stage1_num
            + stage2_num
            + stage3_num
            + stage4_num
            + stage5_num,
            :,
        ].reshape(-1, stage5_num, self.feature_single_num)
        x6 = x[
            :,
            stage1_num
            + stage2_num
            + stage3_num
            + stage4_num
            + stage5_num : stage1_num
            + stage2_num
            + stage3_num
            + stage4_num
            + stage5_num
            + stage6_num,
            :,
        ].reshape(-1, stage6_num, self.feature_single_num)
        feature_embed = [x1, x2, x3, x4, x5, x6]
        rnn_lists = [self.rnn1, self.rnn2, self.rnn3, self.rnn4, self.rnn5, self.rnn6]
        stage_nums = [stage1_num, stage2_num, stage3_num, stage4_num, stage5_num, stage6_num]
        outs = []

        for xx, rnn_list, stage_num in zip(feature_embed, rnn_lists, stage_nums):
            out = self.rnn_worker(rnn_list, xx, stage_num)
            outs.append(out)

        out_final = torch.cat(outs, dim=1)
        out_final = self.embedding1(out_final)
        out2 = self.embedding2(y[:, self.feature_total_num_array])
        out_final = torch.cat([out_final, out2], dim=1)
        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final


class Predict_harsbnn_cifar(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device, layer_len_max=32):
        super(Predict_harsbnn_cifar, self).__init__()
        self.hidden_size = 128
        self.device = device
        self.feature_single_num = 8
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]
        self.embedding = nn.Linear(self.feature_single_num, embed_dim)
        self.rnn1 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(8 + 1)
            ]
        )
        self.rnn2 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(8 + 1)
            ]
        )
        self.rnn3 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(8 + 1)
            ]
        )
        self.rnn4 = nn.ModuleList(
            [
                nn.LSTM(self.feature_single_num, self.hidden_size, dropout=dropout, batch_first=True)
                for i in range(8 + 1)
            ]
        )
        self.embedding1 = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size * 2, 1)
        )

    def rnn_worker(self, rnn_list, feature_embed, stage_num):
        layer_index = 0
        for run_single in rnn_list:
            if layer_index == 0:
                out, (hn, cn) = run_single(feature_embed[:, layer_index, :])  # (batch, len, hidden)
            elif layer_index < stage_num:
                out, (hn, cn) = run_single(feature_embed[:, layer_index, :], (hn, cn))
            else:
                break
            layer_index = layer_index + 1
        return out

    def forward(self, x):
        x = x.to(self.device)

        batch, len, vec = x.shape
        y = x[:, -1, :]
        stage1_num = int(y[:, 5].item())
        stage2_num = int(y[:, 6].item())
        stage3_num = int(y[:, 7].item())
        stage4_num = int(y[:, 8].item())
        # print(f"stage1_num:{stage1_num}")
        # print(f"stage2_num:{stage2_num}")
        # print(f"stage3_num:{stage3_num}")
        # print(f"stage4_num:{stage4_num}")

        x = x[:, :, : self.feature_single_num]
        x1 = x[:, :stage1_num, :].reshape(-1, stage1_num, self.feature_single_num)
        x2 = x[:, stage1_num : stage1_num + stage2_num, :].reshape(-1, stage2_num, self.feature_single_num)
        x3 = x[:, stage1_num + stage2_num : stage1_num + stage2_num + stage3_num, :].reshape(
            -1, stage3_num, self.feature_single_num
        )
        x4 = x[:, stage1_num + stage2_num + stage3_num : stage1_num + stage2_num + stage3_num + stage4_num, :].reshape(
            -1, stage4_num, self.feature_single_num
        )
        feature_embed = [x1, x2, x3, x4]
        rnn_lists = [self.rnn1, self.rnn2, self.rnn3, self.rnn4]
        stage_nums = [stage1_num, stage2_num, stage3_num, stage4_num]
        outs = []

        for xx, rnn_list, stage_num in zip(feature_embed, rnn_lists, stage_nums):
            out = self.rnn_worker(rnn_list, xx, stage_num)
            outs.append(out)

        out_final = torch.cat(outs, dim=1)
        out_final = self.embedding1(out_final)
        out2 = self.embedding2(y[:, self.feature_total_num_array])
        out_final = torch.cat([out_final, out2], dim=1)
        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final


class Predict_vSVR(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout, device):
        super(Predict_vSVR, self).__init__()
        self.hidden_size = 256
        self.device = device
        self.feature_single_num = 8
        # self.feature_single_num = 4
        # self.feature_single_num = 6
        self.feature_total_num = 4
        self.feature_total_num_array = [0, 1, 2, 3]

        self.embedding2 = nn.Sequential(
            nn.Linear(self.feature_total_num, self.hidden_size),
            LearnableScale(self.hidden_size),
            LearnableBias(self.hidden_size),
            nn.PReLU(),
        )

        self.linear = nn.Sequential(
            # nn.Sigmoid(),
            # 第 1 种
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        y = x[:, -1, :]
        out2 = self.embedding2(y[:, self.feature_total_num_array])

        out_final = out2

        out_final = self.linear(out_final)
        out_final = out_final.view(-1)

        return out_final
