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

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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
  print(model)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = KLDCE()
  criterion = criterion.cuda()

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  train_transform, valid_transform = utils._data_transforms_svhn(args)
  train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
  valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)

  #scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30], gamma=0.1, last_epoch=-1)#Densenet
  #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    train_obj = train(train_queue, model, criterion, optimizer)
    #scheduler.step()
    logging.info('train_loss %f', train_obj)

    infer(valid_queue, model, criterion,epoch)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    r,mean,log_var = model(input)
    loss = criterion(input,r,log_var,mean)
    loss.backward()
    optimizer.step()

    n = input.size(0)
    objs.update(loss.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %ef', step, objs.avg)

  return objs.avg


def infer(valid_queue, model, criterion,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()
  
      r,mean,log_var= model(input)
    break

  img=r.detach().cpu()
  img=torchvision.utils.make_grid(img)
  img=img.permute(1,2,0)
  plt.imshow(img)
  plt.savefig(args.save+'/'+str(epoch)+'.png')


if __name__ == '__main__':
  main() 

