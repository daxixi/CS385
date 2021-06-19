import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import MobileNet

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sklearn.externals
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=256, help='gpu device id')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')

args = parser.parse_args()

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  train_transform, valid_transform = utils._data_transforms_svhn(args)
  train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
  valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model = MobileNet.get_model()
  utils.load(model,'weights.pt')
  model = model.cuda()
  train_dataset=[]
  train_labels=[]
  valid_dataset=[]
  valid_labels=[]

  ct=0
  for input,target in valid_queue:
    #input = Variable(input).cuda()
    #_,input = model(input)
    #input=input.detach().cpu()
    ct+=1
    if ct>10:
      break
    input=input.view(input.size(0),-1)
    for i in range(input.shape[0]):
      valid_dataset.append(input[i].numpy().tolist())
      valid_labels.append(target[i])
  with open('eval-sigmoid-20210603-102031/model.pickle','rb') as f:
    clf=pickle.load(f)
  SV=clf.support_vectors_.tolist()
  valid_dataset+=SV
  for i in range(len(SV)):
    valid_labels.append(-1)
  valid_dataset=np.array(valid_dataset)
  valid_labels=np.array(valid_labels)

  pca=PCA(n_components=2)
  new=pca.fit_transform(valid_dataset)
  print(pca.explained_variance_ratio_)

  pos_x=[]
  pos_y=[]
  neg_x=[]
  neg_y=[]
  s_x=[]
  s_y=[]
  for step,n in enumerate(new):
    if valid_labels[step]==9:
      pos_x.append(n[0])
      pos_y.append(n[1])
    elif valid_labels[step]==-1:
      s_x.append(n[0])
      s_y.append(n[1])
    else:
      neg_x.append(n[0])
      neg_y.append(n[1])

  plt.scatter(neg_x,neg_y,color='blue',label='neg')
  plt.scatter(pos_x,pos_y,color='red',label='pos')
  plt.scatter(s_x,s_y,color='black',s=1,label='support vectors')
  plt.legend()
  plt.show()

  #with open('eval-linearext-20210602-205849/model.pickle','rb') as f:
  #  clf=pickle.load(f)
  #logits=clf.predict(valid_dataset)
  #acc=accuracy_score(valid_labels,logits)
  #print(acc)


if __name__ == '__main__':
  main() 

