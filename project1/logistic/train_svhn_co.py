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
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=256, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='gpu device id')
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

class LR(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight=nn.Parameter(torch.Tensor(3073,10))
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

  model=LR()
  model=model.cuda()

  optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  for epoch in range(args.epochs):
    train_acc, train_obj= train(train_queue, model, criterion, optimizer)
    logging.info('epoch %d train_acc %f', epoch,train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('epoch %d valid_acc %f', epoch,valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input=input.view(input.size(0),-1)
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits = model(input)
    sigmoid=nn.Sigmoid()
    logits=sigmoid(logits)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()
    
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(),n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % 50 == 0:
      logging.info('train %03d %f %f %f', step, objs.avg,top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input=input.view(input.size(0),-1)
    with torch.no_grad():
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()
          
      logits = model(input)
      sigmoid=nn.Sigmoid()
      logits=sigmoid(logits)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % 50 == 0:
      logging.info('valid %03d %f %f', step,top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main() 

