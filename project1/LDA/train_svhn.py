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

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=256, help='gpu device id')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def train_LDA(train_dataset,train_label,cls):
  pos=[]
  neg=[]
  for i,l in zip(train_dataset.tolist(),train_label.tolist()):
    if l==cls:
      pos.append(i)
    else:
      neg.append(i)
  neg=torch.tensor(neg).cuda()
  pos=torch.tensor(pos).cuda()
  mupos=torch.mean(pos,dim=0)
  muneg=torch.mean(neg,dim=0)
  sigmapos=pos-mupos.repeat(pos.shape[0],1)
  sigmapos=torch.mm(sigmapos.t(),sigmapos)
  sigmaneg=neg-muneg.repeat(neg.shape[0],1)
  sigmaneg=torch.mm(sigmaneg.t(),sigmaneg)
  sw=sigmapos+sigmaneg
  mu=mupos-muneg
  mu=torch.reshape(mu,(3072,1))
  sb=torch.mm(mu,mu.t())
  A=torch.mm(sw.t(),sb)
  (evals,evecs)=torch.eig(A,eigenvectors=True)
  value,row=torch.abs(evals).max(1)
  value_max,col=value.max(0)
  evals=evals[:,col]
  print(torch.max(evals)/torch.sum(evals))
  evecs=evecs[:,torch.argmax(evals)]
  evecs=torch.reshape(evecs,(3072,1))
  return evecs
  
  

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

  train_dataset=[]
  train_labels=[]
  valid_dataset=[]
  valid_labels=[]
  for input,target in train_queue:
    input=input.view(input.size(0),-1)
    for i in range(input.shape[0]):
      train_dataset.append(input[i].numpy().tolist())
      train_labels.append((target[i]>4)*1)
  train_dataset=np.array(train_dataset)
  train_labels=np.array(train_labels)

  for input,target in valid_queue:
    input=input.view(input.size(0),-1)
    for i in range(input.shape[0]):
      valid_dataset.append(input[i].numpy().tolist())
      valid_labels.append((target[i]>4)*1)
  valid_dataset=np.array(valid_dataset)
  valid_labels=np.array(valid_labels)
  
  models=[]

  for i in range(1):
    lda=train_LDA(train_dataset,train_labels,i)
    models.append(lda)
  
  valid_dataset=torch.from_numpy(valid_dataset).cuda()
  valid_labels=torch.from_numpy(valid_labels)
  predict=torch.mm(valid_dataset.double(),models[0].double())
  predict=(predict>0).int()
  acc=accuracy_score(valid_labels.numpy(),predict.detach().cpu().numpy())
  #predicts=torch.zeros((valid_dataset.shape[0],10))
  #for step,model in enumerate(models):
  #  predict=torch.mm(valid_dataset.double(),model.double())
  #  predicts[:,step]=predict.view(predict.size(0)).detach().cpu()

  #top1 = utils.AvgrageMeter()
  #top5 = utils.AvgrageMeter()
  #prec1, prec5 = utils.accuracy(predicts,valid_labels,topk=(1, 5))
  #n = input.size(0)
  #top1.update(prec1.item(), n)
  #top5.update(prec5.item(), n)

  print(acc)
  #logging.info('train %f %f',top1.avg, top5.avg)


  for cls,model in enumerate(models):
    model=model.cpu()
    model=model.numpy()
    with open(args.save+'/model'+str(cls)+'.pickle','wb') as f:
      pickle.dump(model,f)
  

if __name__ == '__main__':
  main() 

