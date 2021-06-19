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
#import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch.autograd import Variable
from simpleVAE import VAE,CVAE
from Losses import KLDCE

matplotlib.use('Agg')

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

args = parser.parse_args()

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  model=CVAE(1024)
  utils.load(model,'eval-latent1024-20210611-113912/weights.pt')
  print(model)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  train_transform, valid_transform = utils._data_transforms_svhn(args)
  train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
  valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)

  for epoch in range(1):
    valid_acc = infer(valid_queue, model)
    print('valid_acc %f', valid_acc)


def infer(valid_queue, model):
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  features=None
  labels=None
  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()
  
      logits= model.encoder(input)
      logits=logits.view(logits.size(0),-1)

      #prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      #n = input.size(0)
      #top1.update(prec1.item(), n)
      #top5.update(prec5.item(), n)

      if features==None:
        features=logits.detach().cpu()
      else:
          features=torch.cat((features,logits.detach().cpu()),dim=0)
      if labels==None:
          labels=target.detach().cpu()
      else:
          labels=torch.cat((labels,target.detach().cpu()),dim=0)
      if features.shape[0]>1000:
          break

  pca=PCA(n_components=10)
  features=pca.fit_transform(features.numpy())
  tsne=TSNE(n_components=2)
  features=tsne.fit_transform(features)
  for cl in range(10):
      indices=np.where(labels==cl)
      plt.scatter(features[indices,0],features[indices,1],label=cl)
  plt.legend()
  plt.savefig('VAEpng')

  return 0#top1.avg

if __name__ == '__main__':
  main() 

