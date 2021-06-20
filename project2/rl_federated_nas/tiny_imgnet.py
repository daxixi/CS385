"""set the working dir"""
import os
import sys

cwd = os.getcwd()
pwd = cwd[:cwd.rfind('/')]
sys.path.append(pwd)

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

_train_dir = "/home/lxiang_stu/Tiny-ImageNet-Classifier/images/64/train"
_val_dir = "/home/lxiang_stu/Tiny-ImageNet-Classifier/images/64/val"

def create_train_dataset(width=224, batch_size=64, shuffle=True, num_workers=4):
    print('Creating Training dataset...')
    transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(width),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                                 ])
    trainset = ImageFolder(_train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    return trainloader



def create_test_dataset(width=224, batch_size=64, shuffle=True, num_workers=4):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
            transforms.Resize(width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                                 ])
    testset = ImageFolder(_val_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    return testloader

def get_loader(bc=32):
    return ( create_train_dataset(224,bc,True,4),#train
             create_train_dataset(224,bc,True,4),#cal
             create_test_dataset(224,bc,False,4),#val
             create_test_dataset(224,bc,True,4)#test
             )
