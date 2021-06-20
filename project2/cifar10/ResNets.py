import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import math

from collections import deque

class Classifier(nn.Module):
    def __init__(self, expansion, num_classes):
        super(Classifier, self).__init__()
        
        self.classifier=nn.Linear(512*expansion, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False,padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False,padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
          self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

  def __init__(self, block, layer_blocks, num_classes=10,start_channel=3,width=1):
    super().__init__()

    self.num_classes = num_classes

    self.layers = []

    self.prep = nn.Sequential(nn.Conv2d(start_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))

    list_planes = [64, 128, 256, 512]
    list_planes = [x*width for x in list_planes]
    list_stride = [1, 2, 2, 2]
    self.inplanes = 64
    
    stage = 'stage'
    for i, planes in enumerate(list_planes):
      layers=[]
      strides = [list_stride[i],] + [1,]*(layer_blocks[i]-1)
      cur_stage = stage + '_' + str(i+1)
      for j,stride in enumerate(strides):
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        setattr(self, cur_stage+'_'+str(j+1), layers[-1])
      self.layers.append(nn.Sequential(*layers))

    self.layers1=self.layers[0]
    self.layers2=self.layers[1]
    self.layers3=self.layers[2]
    self.layers4=self.layers[3]
    
    self.avg=nn.AvgPool2d(4)

    self.classifier=Classifier(block.expansion*width, num_classes)

  def forward(self,x):
      x=self.prep(x)
      x=self.layers1(x)
      x=self.layers2(x)
      x=self.layers3(x)
      x=self.layers4(x)
      x=self.avg(x)
      x=self.classifier(x)
      return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = CifarResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
def resnet50_2(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   width=2)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
