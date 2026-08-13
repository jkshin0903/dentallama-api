"""HRNet-W32 for cephalometric landmark detection (Stage 1).

Self-contained, trained from scratch on Aariz/ISBI. Input: (B, 1, H, W) grayscale. Output: (B, 29,
192, 192) landmark heatmaps. Architecture matches the standard HRNet-W32 keypoint head used in the
original CephStruct Stage 1 so weights and behavior are reproducible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, i, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[i] != num_channels[i] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[i], num_channels[i] * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(num_channels[i] * block.expansion),
            )
        layers = [block(self.num_inchannels[i], num_channels[i], stride, downsample)]
        self.num_inchannels[i] = num_channels[i] * block.expansion
        for _ in range(1, num_blocks[i]):
            layers.append(block(self.num_inchannels[i], num_channels[i]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList(
            [self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)]
        )

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        nb, nic = self.num_branches, self.num_inchannels
        fuse_layers = []
        for i in range(nb if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(nb):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(nic[j], nic[i], 1, 1, 0, bias=False), nn.BatchNorm2d(nic[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        out_c = nic[i] if k == i - j - 1 else nic[j]
                        seq = [nn.Conv2d(nic[j], out_c, 3, 2, 1, bias=False), nn.BatchNorm2d(out_c)]
                        if k != i - j - 1:
                            seq.append(nn.ReLU(inplace=True))
                        convs.append(nn.Sequential(*seq))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
                                          size=[x[i].shape[-2], x[i].shape[-1]],
                                          mode="bilinear", align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """HRNet-W32 producing `num_landmarks` heatmaps at 192x192."""

    def __init__(self, num_landmarks: int = 29):
        super().__init__()
        self.num_joints = num_landmarks
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        c = [32 * BasicBlock.expansion, 64 * BasicBlock.expansion]
        self.transition1 = self._make_transition_layer([256], c)
        self.stage2, pre = self._make_stage(
            dict(NUM_MODULES=1, NUM_BRANCHES=2, NUM_BLOCKS=[4, 4], NUM_CHANNELS=[32, 64]), c)

        c = [32, 64, 128]
        self.transition2 = self._make_transition_layer(pre, c)
        self.stage3, pre = self._make_stage(
            dict(NUM_MODULES=4, NUM_BRANCHES=3, NUM_BLOCKS=[4, 4, 4], NUM_CHANNELS=[32, 64, 128]), c)

        c = [32, 64, 128, 256]
        self.transition3 = self._make_transition_layer(pre, c)
        self.stage4, pre = self._make_stage(
            dict(NUM_MODULES=3, NUM_BRANCHES=4, NUM_BLOCKS=[4, 4, 4, 4],
                 NUM_CHANNELS=[32, 64, 128, 256]), c, multi_scale_output=True)

        self.final_layer = nn.Conv2d(pre[0], self.num_joints, 1, 1, 0)
        self.dropout = nn.Dropout2d(0.1)
        self._initialize_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, pre, cur):
        nb_cur, nb_pre, layers = len(cur), len(pre), []
        for i in range(nb_cur):
            if i < nb_pre:
                if cur[i] != pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre[i], cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cur[i]), nn.ReLU(inplace=True)))
                else:
                    layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - nb_pre):
                    inc = pre[-1]
                    outc = cur[i] if j == i - nb_pre else inc
                    convs.append(nn.Sequential(
                        nn.Conv2d(inc, outc, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outc), nn.ReLU(inplace=True)))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    def _make_stage(self, cfg, num_inchannels, multi_scale_output=True):
        block = BasicBlock
        num_channels = [c * block.expansion for c in cfg["NUM_CHANNELS"]]
        # transition feeds branch channels already; modules operate on `num_channels`.
        modules = []
        nic = num_channels
        for i in range(cfg["NUM_MODULES"]):
            reset = multi_scale_output or i < cfg["NUM_MODULES"] - 1
            m = HighResolutionModule(cfg["NUM_BRANCHES"], block, cfg["NUM_BLOCKS"], nic,
                                     cfg["NUM_CHANNELS"], "SUM", reset)
            modules.append(m)
            nic = m.get_num_inchannels()
        return nn.Sequential(*modules), nic

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [t(x) if t is not None else x for t in self.transition1]
        y = self.stage2(x_list)

        x_list = [self.transition2[i](y[-1]) if self.transition2[i] is not None else y[i]
                  for i in range(len(self.transition2))]
        y = self.stage3(x_list)

        x_list = [self.transition3[i](y[-1]) if self.transition3[i] is not None else y[i]
                  for i in range(len(self.transition3))]
        y = self.stage4(x_list)

        x = self.final_layer(self.dropout(y[0]))
        return F.interpolate(x, size=(192, 192), mode="bilinear", align_corners=False)
