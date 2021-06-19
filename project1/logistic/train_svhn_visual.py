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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sklearn.externals
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=256, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='gpu device id')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

CIFAR_CLASSES = 10

class LR(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight=nn.Parameter(torch.Tensor(3073,1))
    torch.nn.init.normal_(self.weight,mean=0,std=0.01)

  def forward(self,x):
    bias=torch.ones((x.shape[0],1)).cuda()
    x=torch.cat((x,bias),1)
    x=torch.mm(x,self.weight)
    return x
  
class logistic_loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x,y):
    x=x.view(x.size(0))
    e1=-torch.mul(x,y)
    e2=torch.log(1+e1.exp())
    return torch.mean(e2)

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

  models=[LR() for _ in range(10)] #use 10 binary classifiers
  for i,model in enumerate(models):
    utils.load(model,'eval-logitstic_ridge_lambda10-20210604-141720/weights_'+str(i)+'.pt') 
  models=[model.cuda() for model in models]

  infer(valid_queue, models)
  
def infer(valid_queue, models):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  for model in models:
    model.eval()

  model_map=[]
  for model in models:
    heatmap=model.weight[:3072].detach().cpu().numpy().reshape((3,32,32))
    heatmap=np.mean(heatmap,axis=0)
    heatmap=np.maximum(heatmap,0)
    heatmap/=np.max(heatmap)
    model_map.append(heatmap)

  choose_img={}
  for step, (input, target) in enumerate(valid_queue):
    for img,t in zip(input,target):
      if not t.item() in choose_img:
        choose_img[t.item()]=img
      if len(choose_img)==10:
        break
    if len(choose_img)==10:
      break
    
  for i in range(10):
    img=choose_img[i]
    heatmap=model_map[i]
    img=img.numpy().transpose((1,2,0))
    img=np.uint8(255*img)
    heatmap=np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img=cv2.resize(superimposed_img,(224,224))
    cv2.imwrite(str(i)+'.png',superimposed_img)

if __name__ == '__main__':
  main() 

