import torch
import torchvision

net=torchvision.models.MobileNetV2()
parm=net.parameters()
for p in parm:
    print(p.grad)