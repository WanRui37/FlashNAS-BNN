import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch.cuda.amp import autocast as autocast

sys.path.append("./train/nets/")
from .dynamic_operations import (
    DynamicBatchNorm2d,
    DynamicBinConv2d,
    DynamicFPLinear,
    DynamicLearnableBias,
    DynamicPReLU,
    DynamicQConv2d,
)

STAGE_OUT_CHANNEL = os.environ.get("STAGE_OUT_CHANNEL")
CONV_TYPE_ARRAY = os.environ.get("CONV_TYPE_ARRAY")
STRIDE_ARRAY = os.environ.get("STRIDE_ARRAY")
ARCH_ALL = os.environ.get("ARCH_ALL")

if ARCH_ALL is not None:
    arch_all_array = [int(item.strip()) for item in ARCH_ALL[1:-1].split(",")]
    array_length = len(arch_all_array) // 3
    stage_out_channel = arch_all_array[0:array_length]
    stride_array = arch_all_array[array_length : array_length * 2]
    conv_type_array = arch_all_array[array_length * 2 :]
elif STAGE_OUT_CHANNEL is not None:
    stage_out_channel = [int(item.strip()) for item in STAGE_OUT_CHANNEL[1:-1].split(",")]
    conv_type_array = [int(item.strip()) for item in CONV_TYPE_ARRAY[1:-1].split(",")]
    stride_array = [int(item.strip()) for item in STRIDE_ARRAY[1:-1].split(",")]
else:
    stage_out_channel = [
        16,
        16,
        16,
        48,
        48,
        48,
        48,
        48,
        48,
        48,
        128,
        128,
        128,
        128,
        192,
        192,
        192,
        512,
        512,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
        640,
    ]
    stride_array = [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    conv_type_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]

skip_config = [True] + [True] * 1 + [True] * 2 + [True] * 6 + [True] * 2
concat_type_array = [0] * len(stage_out_channel)

print(f"stage_out_channel:{stage_out_channel}")
print(f"conv_type_array:{conv_type_array}")
print(f"stride_array:{stride_array}")


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(
            torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True
        )
        # print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class adaptive_add(nn.Module):
    def __init__(self, inp, oup):
        super(adaptive_add, self).__init__()
        self.move = LearnableBias(oup)
        self.inp = inp
        self.oup = oup

    def forward(self, src, residual):
        if self.inp == self.oup:
            out = src + residual
        else:
            residual = torch.cat([residual for _ in range(round(self.oup / self.inp + 0.5))], dim=1)[:, : self.oup]
            out = src + self.move(residual)
        return out


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride, w_bit=8, a_bit=None, channel_wise=True):
        super(firstconv3x3, self).__init__()
        # print(f"oup:{oup}")
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.w_bit = w_bit
        self.oup = oup
        self.inp = inp
        self.padding = (3 - 1) // 2
        self.stride = stride
        shape = (self.oup, self.inp, 3, 3)
        self.register_parameter("weight", nn.Parameter(torch.rand(shape)))

        if w_bit != 32:
            if channel_wise:
                self.scale_channel = oup
            else:
                self.scale_channel = 1
            self.register_parameter("w_scale", nn.Parameter(torch.ones(self.scale_channel)))
            self.register_buffer("w_init_state", torch.zeros(1))
            self.w_min_q = -1 * 2 ** (w_bit - 1)
            self.w_max_q = 2 ** (w_bit - 1) - 1

        if a_bit != 32 and a_bit is not None:
            self.register_parameter("a_scale", nn.Parameter(torch.ones(1)))
            self.register_parameter("a_zero_point", nn.Parameter(torch.ones(1)))
            self.register_buffer("a_init_state", torch.zeros(1))
            self.a_min_q = 0
            self.a_max_q = 2**a_bit - 1
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        if self.w_bit != 32:
            if self.training and self.w_init_state == 0:
                self.w_init_state.fill_(1)
                w_detach = self.weight.detach()
                if self.scale_channel == 1:
                    w_mean = w_detach.mean()
                    w_std = w_detach.std()
                else:
                    dim = [i for i in range(1, w_detach.dim())]
                    w_mean = w_detach.mean(dim=dim)
                    w_std = w_detach.std(dim=dim)
                v1 = torch.abs(w_mean - 3 * w_std)
                v2 = torch.abs(w_mean + 3 * w_std)
                self.w_scale.data.copy_(torch.max(v1, v2) / 2 ** (self.w_bit - 1))
            self.w_scale.data.clamp_(self.eps)
            g = 1.0 / math.sqrt(self.weight.numel() * self.w_max_q)
            cur_w_scale = grad_scale(self.w_scale, g)
            if self.scale_channel != 1:
                cur_w_scale = cur_w_scale.view(self.oup, 1, 1, 1)
            weight = round_pass(self.weight / cur_w_scale).clamp(self.w_min_q, self.w_max_q) * cur_w_scale

        out = F.conv2d(x, weight, None, stride=self.stride, padding=self.padding)
        out = self.bn1(out)

        return out


class BinaryActivation(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors

        grad_input_wrt_output2 = torch.zeros_like(grad_output)
        ge0_lt1_mask = input.ge(0) & input.lt(1)
        grad_input_wrt_output2 = torch.where(ge0_lt1_mask, (2 - 2 * input), grad_input_wrt_output2)
        gen1_lt0_mask = input.ge(-1) & input.lt(0)
        grad_input_wrt_output2 = torch.where(gen1_lt0_mask, (2 + 2 * input), grad_input_wrt_output2)
        grad_input = grad_input_wrt_output2 * grad_output

        return grad_input


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BasicBlock_Single(nn.Module):
    def __init__(self, inplanes, planes, stride=1, conv_type=0, concat_type=0):
        super(BasicBlock_Single, self).__init__()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.conv_type = conv_type
        self.concat_type = concat_type

        if self.inplanes == self.planes and concat_type == 1:
            print("concat_type setting error!!!")
            exit(0)

        norm_layer = nn.BatchNorm2d
        self.move11 = LearnableBias(inplanes)

        if conv_type == 0:
            self.binary_conv = binaryconv3x3(inplanes, planes, stride=stride)
        elif conv_type == 1:
            self.binary_conv = binaryconv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)

        self.adaptive = adaptive_add(inplanes, planes)
        self.move12 = LearnableBias(planes)
        self.prelu1 = nn.PReLU(planes)
        self.move13 = LearnableBias(planes)

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move11(x)
        out1 = BinaryActivation.apply(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = self.binary_conv(out1)
        out1 = self.bn1(out1)
        out1 = self.adaptive(out1, x)

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        return out1


class reactnet(nn.Module):
    def __init__(self, num_classes=100):
        super(reactnet, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            else:
                self.feature.append(
                    BasicBlock_Single(
                        stage_out_channel[i - 1],
                        stage_out_channel[i],
                        stride_array[i],
                        conv_type_array[i],
                        concat_type_array[i],
                    )
                )
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_out_channel[-1], num_classes)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            # if i==0:
            #     x, fm = block(x)
            # else:
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  # , fm


def reactnet_firstBConv_lbp_ther_48c_100_stage2(**kwargs):
    model = reactnet(**kwargs)
    return model


def flashnas_bnn_stage2(**kwargs):
    model = reactnet(**kwargs)
    return model


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = ReActNet(num_classes=1000)
    print(model(x))
