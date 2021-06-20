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
import random
from torch.autograd import Variable

num_class = 10
num_user = 10
user_class = 5

pool = list(i for i in range(num_class))
flag = {}
class_user = num_user * user_class // num_class
print("class user",class_user)
user_split = [[] for _ in range(num_user)]
# print(class_user)
# for ele in pool:
#     flag[ele] = class_user
# print(flag)
# for i in range(num_user):
#     user_split = random.sample(pool,user_class)
#     print("user",i,user_split)
#     for ele in user_split:
#         flag[ele] -= 1
#         if flag[ele] == 0:
#             pool.remove(ele)
#     print("pool",pool)
#     print("flag",flag)


for i in range(num_class):
    idx = i
    for j in range(idx,idx+class_user):
        if j >= num_user:
            j -= num_user
        user_split[j].append(i)

for i in range(num_user):
    print(user_split[i])




