import torch
import torchvision.datasets as dset
from torchvision.transforms import transforms
import numpy as np
import random


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def even_split(dataset, num_user, num_class=10):
    category = [[] for i in range(num_class)]
    for (img, label) in dataset:
        category[label].append(img)
    user_split = [[] for i in range(num_user)]
    split = len(category[0]) // num_user
    for l in range(num_class):
        for u in range(num_user):
            for img in category[l][u*split:(u+1)*split]:
                user_split[u].append((img, l))
    for u in range(num_user):
        random.shuffle(user_split[u])
    return user_split

def none_iid_split(dataset, num_user=10, user_class=5, num_class=10):
    assert num_user==10, "non iid dataset works only for 10 clients"
    category = [[] for i in range(num_class)]
    for (img, label) in dataset:
        category[label].append(img)
    user_split = [[] for _ in range(num_user)]
    # partition for 10 clients
    user_partition = [[0, 6, 7, 8, 9],[0, 1, 7, 8, 9],[0, 1, 2, 8, 9],[0, 1, 2, 3, 9],[0, 1, 2, 3, 4],[1, 2, 3, 4, 5],[2, 3, 4, 5, 6],[3, 4, 5, 6, 7],[4, 5, 6, 7, 8],[5, 6, 7, 8, 9]]
    class_idx = [0 for _ in range(num_class)]
    class_len = [len(category[i])//(num_class*user_class//num_user) for i in range(num_class)]
    for i in range(num_user):
        user_part = user_partition[i]
        for label in user_part:
            class_start = class_idx[label]
            class_size = class_len[label]
            class_idx[label] += 1
            for j in range(class_start*class_size,class_size*(class_start+1)):
                img = category[label][j]
                user_split[i].append((img,label))
    for u in range(num_user):
        random.shuffle(user_split[u])
    return user_split

if __name__ == '__main__':
    train_transform, valid_transform = _data_transforms_cifar10()
    dataset = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    user_split = even_split(dataset, 10)
    for i in range(10):
        train_data = user_split[i]
        print(len(train_data))


    for i in range(10):
        train_data = user_split[i]

        num_train = len(train_data)
        indices = list(range(num_train))


        train_queue = torch.utils.data.DataLoader(
              train_data, batch_size=32,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
              pin_memory=True, num_workers=2)


        for step, (data, target) in enumerate(train_queue):
            print(target)
            break


