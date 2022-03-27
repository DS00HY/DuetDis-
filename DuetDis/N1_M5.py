# -*- coding: utf-8 -*-
"""

"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

######
def elu():
    return nn.ELU(inplace=True)

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

def elu():
    return nn.ELU(inplace=True)

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        #self.relu = nn.ReLU(inplanes=True)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        #self.se = Selayer(planes * 4)
        self.droprate = 0.2
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        #out = self.se(out)

        out += residual
        out = self.gelu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilations=1):
        super(Bottleneck, self).__init__()
        self.dilations=dilations
        kernel_size_1 = 1
        kernel_size_2 = 3
        kernel_size_3 = 1
        padding1 = dilations * (kernel_size_1 - 1) // 2
        padding2 = dilations * (kernel_size_2 - 1) // 2
        padding3 = dilations * (kernel_size_3 - 1) // 2
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size_1,padding=padding1, dilation=dilations, bias=True)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size_2, stride=1, padding=padding2, dilation=dilations, bias=True)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=kernel_size_3, padding=padding3, dilation=dilations, bias=True)
        self.bn3 = nn.InstanceNorm2d(self.expansion)
        self.downsample = downsample
        self.gelu = nn.GELU()
        self.droprate = 0.0
        self.stride = stride
        self.excitation1 = nn.Conv2d(planes*self.expansion, planes*self.expansion // 16, kernel_size=1)
        self.excitation2 = nn.Conv2d(planes*self.expansion // 16, planes*self.expansion, kernel_size=1)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.gelu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training) 
        #######
        fm_size = out.size()[2]
        scale_weight = F.avg_pool2d(out, fm_size)
        scale_weight = F.gelu(self.excitation1(scale_weight))
        scale_weight = F.sigmoid(self.excitation2(scale_weight))
        out = out * scale_weight.expand_as(out)
        #######
        #out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.gelu(out)
        
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        self.planes = 128
        self.expansion = 4
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(151, 64, kernel_size=1,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.gelu = nn.GELU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer_2(block, 128, layers[2])
        self.layer4 = self._make_layer(block, 128, layers[3])

        self.to_theta = nn.Conv2d(self.inplanes, 25, kernel_size=1)
        self.to_phi = nn.Conv2d(self.inplanes, 13, kernel_size=1)
        self.to_distance = nn.Conv2d(self.inplanes, 37, kernel_size=1)
        self.to_omega = nn.Conv2d(self.inplanes, 25, kernel_size=1)
        self.lastlayer=nn.Conv2d(self.inplanes,37,3,padding=1)
        self.sig=nn.Sigmoid()
        self.soft=nn.Softmax(1)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.InstanceNorm2d(planes*block.expansion),
        )
        layers = []
        #layers.append(block(self.inplanes, planes, stride, downsample))
        layers.append(block(self.inplanes, planes, stride,downsample, dilations=1))
        d = 1
        self.inplanes = planes * block.expansion
       
        for i in range(1, blocks):
            d =  d * 2 
            layers.append(block(self.inplanes, planes, dilations=d))
        return nn.Sequential(*layers)

    def _make_layer_2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.InstanceNorm2d(planes*block.expansion),
        )
        layers = []
        #layers.append(block(self.inplanes, planes, stride, downsample))
        layers.append(block(self.inplanes, planes, stride, downsample, dilations=1))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            d = i * 2 
            layers.append(block(self.inplanes, planes, dilations=d))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn1(x)
        x = self.gelu(x)
        symmetrized = 0.5*(x+torch.transpose(x, -1, -2))
        theta = self.to_theta(x)
        phi = self.to_phi(x)
        distance = self.to_distance(symmetrized)
        omega = self.to_omega(symmetrized)
        return theta, phi, distance, omega
        return x

    
def resnet46(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 6, 10, 3])
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
def seresnet101(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        pass
    return model

def seresnet152(pretrained=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        pass
    return model
