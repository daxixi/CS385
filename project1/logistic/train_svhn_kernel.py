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

class ridge(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight=nn.Parameter(torch.Tensor(3073,1))
    torch.nn.init.normal_(self.weight,mean=0,std=0.01)

  def forward(self,x,y):
    x=x.view(x.size(0))
    K=torch.zeros((x.shape[0],x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
      for j in range(x.shape[0]):
        K[i][j]=x[i]-x[j]
    K=-torch.sum(K,2)
    K=K.exp()
    
    
    y=y.int()
    bias=torch.ones((x.shape[0],1)).cuda()
    x=torch.cat((x,bias),1)
    x=torch.mm(x,self.weight)
    
    p1=y-x
    p1=torch.norm(p1,p=2)/p1.size(0)
    p1=p1*0.5
    p2=torch.norm(self.weight,p=1)
    lambda_=10
    loss=p1+p2
    return x,loss

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

  models=[ridge() for _ in range(10)] #use 10 binary classifiers
  models=[model.cuda() for model in models]

  optimizers = [torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9) for model in models]

  for epoch in range(args.epochs):
    train_acc, train_obj,models = train(train_queue, models, optimizers)
    logging.info('epoch %d train_acc %f', epoch,train_acc)

    valid_acc, valid_obj = infer(valid_queue, models)
    logging.info('epoch %d valid_acc %f', epoch,valid_acc)

    for i,model in enumerate(models):
      utils.save(model, os.path.join(args.save, 'weights_'+str(i)+'.pt'))


def train(train_queue, models,optimizers):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  for model in models:
    model.train()

  for step, (input, target) in enumerate(train_queue):
    input=input.view(input.size(0),-1)
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    predicts=torch.zeros((target.shape[0],10))
    losses=0
    for cls,(model,optimizer) in enumerate(zip(models,optimizers)):
      optimizer.zero_grad()
      target_cls=(target==cls)
      logits,loss = model(input,target_cls)
      loss.backward()
      optimizer.step()
      sigmoid=nn.Sigmoid()
      logits=sigmoid(logits)
      predicts[:,cls]=logits.detach().cpu()
      losses+=loss
    

    target=target.cpu()
    prec1, prec5 = utils.accuracy(predicts, target, topk=(1, 5))
    n = input.size(0)
    objs.update(losses,n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % 50 == 0:
      logging.info('train %03d %f %f %f', step, objs.avg,top1.avg, top5.avg)

  return top1.avg, objs.avg,models


def infer(valid_queue, models):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  for model in models:
    model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input=input.view(input.size(0),-1)
    with torch.no_grad():
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()
          
      predicts=torch.zeros((target.shape[0],10))
      for cls,model in enumerate(models):
        target_cls=(target==cls)
        logits,loss = model(input,target_cls)
        sigmoid=nn.Sigmoid()
        logits=sigmoid(logits)
        logits=logits.view(logits.size(0))
        predicts[:,cls]=logits.detach().cpu()

    target=target.cpu()
    prec1, prec5 = utils.accuracy(predicts, target, topk=(1, 5))
    n = input.size(0)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % 50 == 0:
      logging.info('valid %03d %f %f', step,top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main() 

