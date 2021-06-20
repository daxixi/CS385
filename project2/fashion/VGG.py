import torch.nn as nn
import torch.nn.functional as F
import torch
#from torchsummary import summary


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        
        self.classifier=nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
    
class Linear(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Linear, self).__init__()

        self.bn = nn.BatchNorm1d(4096)
        self.relu=nn.ReLU()
        self.classifier=nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class VGG(nn.Module):
    """
    VGG builder
    """

    def __init__(self, arch, num_classes=10,start_channel=1) -> object:
        super(VGG, self).__init__()
        self.in_channels = start_channel
        self.layers = []
        for channels,num in zip([64,128,256,512,512],arch):
            uplayers=[]
            for i in range(num):
                layers=[]
                layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
                layers.append(nn.BatchNorm2d(channels))
                layers.append(nn.ReLU())
                self.in_channels = channels
                if i==num-1:
                    layers.append(nn.MaxPool2d(2))    
                module=nn.Sequential(*layers)
                uplayers.append(module)
            self.layers.append(nn.Sequential(*uplayers))
        self.linear1=Linear(1 * 1 * 512, 4096)
        self.linear2=Linear(4096, 4096)
        self.linear3=nn.Linear(4096, num_classes)
        self.block1=self.layers[0]
        self.block2=self.layers[1]
        self.block3=self.layers[2]
        self.block4=self.layers[3]
        self.block5=self.layers[4]

    def forward(self,x):
        x1=self.block1(x)
        x=self.block2(x1)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)

        return x

def VGG_11():
    return VGG([1, 1, 2, 2, 2], num_classes=10,start_channel=1)

def VGG_13():
    return VGG([1, 1, 2, 2, 2], num_classes=10,start_channel=1)

def VGG_16():
    return VGG([2, 2, 3, 3, 3], num_classes=10,start_channel=1)

def VGG_19():
    return VGG([2, 2, 4, 4, 4], num_classes=10,start_channel=1)

def test():
    import torch
    # net = VGG_11()
    # net = VGG_13()
    # net = VGG_16()
    net = VGG_19().cuda()
    x=torch.rand((128,3,32,32)).cuda()
    x=net(x)

if __name__ == '__main__':
    test()
