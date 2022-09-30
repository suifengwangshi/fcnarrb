# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import sys


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class FCN(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=13, channels=4):
        super(FCN, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, channels, kernel_size=3)
        self.blend1 = bn_relu_conv(channels, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        # decode process
        up3 = upsample(skip4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense


class FCNA(nn.Module):
    """FCNA for motif mining"""
    def __init__(self, motiflen=13, channels=4):
        super(FCNA, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, channels, kernel_size=3)
        self.blend1 = bn_relu_conv(channels, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        up5 = self.aap(out1)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense


class RRBlock(nn.Module):
    def __init__(self, nin, nout, size, std=1, shortcut=True):
        super(RRBlock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv1d(nin, nout, size, std, 1),
                                    nn.ReLU(nout),
                                    nn.BatchNorm1d(nout),
                                    nn.Conv1d(nout, nout, size, 1, 1),
                                    nn.BatchNorm1d(nout))
        self.shortcut = shortcut
        self.block2 = nn.Sequential(nn.Conv1d(nin, nout, size, std, 1),
                                    nn.BatchNorm1d(nout))

    def forward(self, data):
        x = data
        out = self.block1(x)

        if self.shortcut:
            out = x + out
        else:
            out = out + self.block2(x)
        return out


class FCNARRB(nn.Module):
    """FCNARRB for motif mining"""
    def __init__(self, motiflen=13, channels=4):
        super(FCNARRB, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # RRBlock for spatial
        self.rrb1 = RRBlock(channels, channels, 3)
        self.rrb2 = RRBlock(64, 64, 3)
        self.rrb3 = RRBlock(64, 64, 3)
        self.rrb4 = RRBlock(64, 64, 3)
        # decode process
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, channels, kernel_size=3)
        self.blend1 = bn_relu_conv(channels, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        up5 = self.aap(out1)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        skip4 = self.rrb4(skip4)
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        skip3 = self.rrb3(skip3)
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        skip2 = self.rrb2(skip2)
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        skip1 = self.rrb1(skip1)
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense
