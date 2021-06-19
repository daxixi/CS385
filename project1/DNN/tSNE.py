import os
import utils
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms as T
import torchvision.datasets as dset

from torch.autograd import Variable
#from model import NetworkCIFAR as Network
import MobileNet
import NiN
import NiN2

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

matplotlib.use('Agg')

def main():
    #model = MobileNet.get_model()
    model= NiN.get_model()
    model_path='eval-NiN-20210531-182047/weights.pt'
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    train_transform, valid_transform = utils._data_transforms_svhn()
    valid_data = dset.SVHN(root='../data', split='test', download=True, transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

    features=None
    labels=None
    for step, (input, target) in enumerate(valid_queue):
        input=input.cuda()
        target=target.cuda()
        output, inner= model(input)
        inner=inner.view(inner.size(0),-1)
        if features==None:
            features=inner.detach().cpu()
        else:
            features=torch.cat((features,inner.detach().cpu()),dim=0)
        if labels==None:
            labels=target.detach().cpu()
        else:
            labels=torch.cat((labels,target.detach().cpu()),dim=0)
        if features.shape[0]>10000:
            break

    #use PCA
    pca=PCA(n_components=10)
    pca.fit(features.numpy())
    #direct TSNE
    features=pca.transform(features.numpy())
    tsne=TSNE(n_components=2)
    features=tsne.fit_transform(features)
    for cl in range(10):
        indices=np.where(labels==cl)
        plt.scatter(features[indices,0],features[indices,1],label=cl)
    plt.legend()
    plt.savefig('resnetfinallayer')
        
main()
