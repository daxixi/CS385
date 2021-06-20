import torch
import numpy as np
import torch.nn.functional as F
import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search import Network
from model_search_local import MaskedNetwork
from architect import Architect

from federated import *




if __name__ == '__main__':
    def test_sample_mask():
        alphas = np.random.random([14,8])
        alphas = torch.Tensor(alphas)
        mask = sample_mask(alphas)
        print(mask)

    from data_distribution import _data_transforms_cifar10
    def test_client_update():
        mask_normal = torch.zeros(14, 8)
        for i in range(14):
            mask_normal[i][4] = 1
        mask_reduce = mask_normal
        criterion = nn.CrossEntropyLoss()
        client_model = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)

        train_transform, valid_transform = _data_transforms_cifar10()
        train_data = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=32,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        updated_model = client_update(train_queue, client_model, criterion)
        cnt = 0
        print("gradient")
        for name, param in updated_model.named_parameters():
            print("name",name)
            print(param.grad)
            cnt += 1
            if cnt > 5:
                break

    def test_fuse_weight_gradient():
        mask_normal = torch.zeros(14, 8)
        for i in range(14):
            mask_normal[i][4] = 1
        mask_reduce = mask_normal
        criterion = nn.CrossEntropyLoss()
        global_model = Network(16, 10, 8, criterion)
        client_model1 = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)
        client_model2 = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)
        init_gradient(global_model)
        init_gradient(client_model1)
        init_gradient(client_model2)
        for n,p in client_model1.named_parameters():
            p.grad = 1 * torch.ones(p.size())
        for n,p in client_model2.named_parameters():
            p.grad = 2 * torch.ones(p.size())
        pos = None
        fuse_weight_gradient(global_model, [client_model1, client_model2])
        for n,p in global_model.named_parameters():
            pos = _extract_pos(n)
            print("pos",pos)
            if pos[2] == 4 or pos[0] in [-1,-2] or pos[1] in [-1,-2]:
                if (p.grad==1.5*torch.ones(p.size())).all():
                    continue
                else:
                    print("check fail",n)
                    print(p.grad)
                    break
            else:
                if (p.grad==torch.zeros(p.size())).all():
                    continue
                else:
                    print("check fail",n)
                    print(p.grad)
                    break
        if pos == (-2,0,0):
            print("check pass")

    def test_fuse_alpha_gradient():
        mask_normal = torch.zeros(14, 8)
        for i in range(14):
            mask_normal[i][4] = 1
        mask_reduce = mask_normal
        criterion = nn.CrossEntropyLoss()
        global_model = Network(16, 10, 8, criterion)
        client_model1 = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)
        client_model2 = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)
        init_gradient(global_model)
        init_gradient(client_model1)
        init_gradient(client_model2)
        client_model1.alphas_normal.grad = 1 * torch.ones(14, 1)
        client_model1.alphas_reduce.grad = 1 * torch.ones(14, 1)
        client_model2.alphas_normal.grad = 2 * torch.ones(14, 1)
        client_model2.alphas_reduce.grad = 2 * torch.ones(14, 1)
        fuse_alphas_gradient(global_model, [client_model1, client_model2])
        print(global_model.alphas_normal.grad)
        print(global_model.alphas_reduce.grad)

    def test_client_param():
        mask_normal = torch.zeros(14, 8)
        for i in range(14):
            mask_normal[i][4] = 1
        mask_reduce = mask_normal
        criterion = nn.CrossEntropyLoss()
        global_model = Network(16, 10, 8, criterion)
        client_model1 = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)
        client_model2 = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)

        client_weight_param(global_model, [client_model1, client_model2])


        global_iter = global_model.named_parameters()
        client_iter1 = client_model1.named_parameters()
        client_iter2 = client_model2.named_parameters()

        for i in range(10):
            gn,gp = next(global_iter)
            cn1,cp1 = next(client_iter1)
            cn2, cp2 = next(client_iter2)
            print('gn',gn)
            print(gp)
            print("cn1",cn1)
            print(cp1)
            print("cn2",cp2)
            print(cp2)

    test_client_param()
    pass