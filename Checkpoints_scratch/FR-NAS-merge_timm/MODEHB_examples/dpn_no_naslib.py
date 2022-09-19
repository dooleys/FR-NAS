from collections import OrderedDict
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from timm.data import IMAGENET_DPN_MEAN, IMAGENET_DPN_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import BatchNormAct2d, ConvNormAct, create_conv2d, create_classifier
from timm.models.registry import register_model
import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
)

from naslib.utils.utils import get_project_root

from primitives import ResNetBasicblock



__all__ = ['DPN']
OP_NAMES = ["ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1", "DilConv5x5","SepConv5x5"]
OP_DICT={"ReLUConvBN3x3": ops.ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
         "ReLUConvBN1x1": ops.ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
         "AvgPool1x1": ops.AvgPool1x1(kernel_size=3, stride=1, affine=False),
         "DilConv5x5": ops.DilConv(C, C, kernel_size=5, stride=1, padding=4, dilation=2, affine=False),
         "SepConv5x5": ops.SepConv(C, C, kernel_size=5, stride=1, padding=2, affine=False)}

class CatBnAct(nn.Module):
    def __init__(self, in_chs, norm_layer=BatchNormAct2d):
        super(CatBnAct, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        return self.bn(x)


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, groups=1, norm_layer=BatchNormAct2d):
        super(BnActConv2d, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, groups=groups)

    def forward(self, x):
        return self.conv(self.bn(x))

class DualPathBlock(AbstractPrimitive):
    def __init__(
        self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False,**kwargs):
        super(DualPathBlock, self).__init__(locals())
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        self.c1x1_w_s1 = None
        self.c1x1_w_s2 = None
        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)

        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc, kernel_size=1)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x,edge_data):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x,edge_data):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        pass

    def forward(self, x,edge_data) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x
        if self.c1x1_w_s1 is None and self.c1x1_w_s2 is None:
            # self.has_proj == False, torchscript requires condition on module == None
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            # self.has_proj == True
            if self.c1x1_w_s1 is not None:
                # self.key_stride = 1
                x_s = self.c1x1_w_s1(x_in)
            else:
                # self.key_stride = 2
                x_s = self.c1x1_w_s2(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        if self.c1x1_c1 is not None:
            # self.b == True, using None check for torchscript compat
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        print(resid.shape)
        return resid,dense
    def get_embedded_ops(self):
        return None
class DiscreteCell(torch.nn.Module):
    def __init__(self,op_list):
      self.op1=OP_DICT[OP_NAMES[op_list[0]]]
      self.op2=OP_DICT[OP_NAMES[op_list[1]]]
      self.op3=OP_DICT[OP_NAMES[op_list[2]]]
      self.op4=OP_DICT[OP_NAMES[op_list[3]]]
      self.op5=OP_DICT[OP_NAMES[op_list[4]]]
      self.op6=OP_DICT[OP_NAMES[op_list[5]]]
    def forward(self,x):
        out2 = self.op1(x)
        out3 = self.op2(x)+self.op4(out2)
        out4 = self.op3(x)+self.op5(out2)+self.op6(out3)
        return out4

class DPN(torch.nn.Module):

    def __init__(self,op_list):
        super().__init__()
        k_sec=[4, 8, 20, 3]
        inc_sec=[20, 64, 64, 128]
        channels=[]
        bw=[]
        r=[]
        b=False
        bw_factor = 4
        k_r=200
        groups=50
        bw1 = 64 * bw_factor
        bw.append(bw1)
        inc = inc_sec[0]
        r1 = (k_r * bw1) // (64 * bw_factor)
        r.append(r1)
        #blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs1 = bw1 + 3 * inc
        channels.append(in_chs1)
        bw2 = 128 * bw_factor
        bw.append(bw2)
        inc = inc_sec[1]
        r2 = (k_r * bw2) // (64 * bw_factor)
        r.append(r2)
        #blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs2 = bw2 + 3 * inc
        channels.append(in_chs2)
        bw3 = 256 * bw_factor
        bw.append(bw3)
        inc = inc_sec[2]
        r3 = (k_r * bw3) // (64 * bw_factor)
        r.append(r3)
        #blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs3 = bw3 + 3 * inc
        channels.append(in_chs3)
        bw4 = 512 * bw_factor
        bw.append(bw4)
        inc = inc_sec[3]
        r4 = (k_r * bw4) // (64 * bw_factor)
        r.append(r4)
        #blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs4 = bw4 + 3 * inc
        channels.append(in_chs4)
        print(channels)
        print(r)
        #
        blocks=[]
        blocks.append(DPN_Stem(num_init_features=128))
        blocks.append(DualPathBlock(128, r[0], r[0], bw[0], inc_sec[0], groups, 'proj', b))
        for _ in range(2, k_sec[0] + 1):
            blocks.append(DiscreteCell(op_list))
        blocks.append(DualPathBlock(channels[0], r[1], r[1], bw[1], inc_sec[1], groups, 'down', b))
        for _ in range(2, k_sec[1] + 1):
            blocks.append(DiscreteCell(op_list))
        blocks.append(DualPathBlock(channels[1], r[2], r[2], bw[2], inc_sec[2], groups, 'down', b))
        for _ in range(2, k_sec[2] + 1):
            blocks.append(DiscreteCell(op_list))
        blocks.append(DualPathBlock(channels[2], r[3], r[3], bw[3], inc_sec[3], groups, 'down', b))
        for _ in range(2, k_sec[3] + 1):
            blocks.append(DiscreteCell(op_list))
        blocks.append(DPN_Tail())
        self.features=torch.nn.Sequential(blocks)
    def forward(self,x):
        return self.features(x)
class DPN_Stem(ops.AbstractPrimitive):
    def __init__(
        self, small=False, num_init_features=64, b=False, output_stride=32, in_chans=3,**kwargs):
        super(DPN_Stem, self).__init__(locals())
        self.b = b
        assert output_stride == 32
        norm_layer = partial(BatchNormAct2d, eps=.001)
        blocks = OrderedDict()
        blocks['conv1_1'] = ConvNormAct(
            in_chans, num_init_features, kernel_size=3 if small else 7, stride=2, norm_layer=norm_layer)
        blocks['conv1_pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(blocks)
    def forward(self, x,edge_data):
        x = self.features(x)
        return x
    def get_embedded_ops(self):
        return None
class DPN_Tail(ops.AbstractPrimitive):
    def __init__(
        self,small=False,num_classes=0,b=False,inc_sec=(16, 32, 24, 128),**kwargs):
        super(DPN_Tail, self).__init__(locals())
        self.b = b
        self.num_classes = num_classes
        bw_factor = 1 if small else 4
        bw = 512 * bw_factor
        inc = inc_sec[3]
        in_chs = bw + 3 * inc
        self.drop_rate = 0.
        print(in_chs)
        fc_act_layer=nn.ELU
        fc_norm_layer = partial(BatchNormAct2d, eps=.001, act_layer=fc_act_layer, inplace=False)
        self.post_cell = CatBnAct(in_chs, norm_layer=fc_norm_layer)
        self.num_features = in_chs
        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        global_pool='avg'
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    def forward(self, x, edge_data):
        pre_logits=False
        print(x.shape)
        x = self.post_cell(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.classifier(x)
            return self.flatten(x)

    def get_embedded_ops(self):
        return None
if __name__ == "__main__":
    network=DPN([0,1,2,3,4,1])
    input=torch.randn([4,3,32,32])
    print(network(input).shape)

