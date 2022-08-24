#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo

from inplace_abn import InPlaceABNSync as BatchNorm2d
resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = BatchNorm2d(out_chan, activation='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = None
        # self.downsample=None
        if in_chan != out_chan or stride != 1:
            self.downsample_conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False)
            self.downsample_bn = BatchNorm2d(out_chan, activation='identity')

            # self.downsample = nn.Sequential(
            #     nn.Conv2d(in_chan, out_chan,
            #               kernel_size=1, stride=stride, bias=False),
            #     BatchNorm2d(out_chan, activation='none'),
            # )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        # if self.downsample is not None:
        #     shortcut = self.downsample(x)
        if self.downsample_conv is not None:
            shortcut = self.downsample_conv(x)
            shortcut = self.downsample_bn(shortcut)

        out = shortcut + residual
        out = self.relu(out)
        return out


# def create_layer_basic(in_chan, out_chan, bnum, stride=1):
#     layers = [BasicBlock(in_chan, out_chan, stride=stride)]
#     for i in range(bnum-1):
#         layers.append(BasicBlock(out_chan, out_chan, stride=1))
#     return nn.Sequential(*layers)

def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.ModuleList(layers)

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        for layer in self.layer1:
            x = layer(x)

        for layer in self.layer2:
            x = layer(x)
        feat8 = x #128

        for layer in self.layer3:
            x = layer(x)
        feat16 = x #256

        for layer in self.layer4:
            x = layer(x)
        # feat32 = x

        return feat8, feat16, x

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict,strict=False)
        # this is called before calling the pretrained model

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (BatchNorm2d, nn.BatchNorm2d)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = Resnet18()
    x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()
