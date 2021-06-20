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
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from model_search_local import MaskedNetwork
from architect import Architect
from federated import sample_mask, client_update, fuse_weight_gradient, init_gradient, client_weight_param, extract_index, uniform_sample_mask
from data_distribution import _data_transforms_cifar10, even_split, none_iid_split
from noniid_svhn import client_data

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--client_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=6000, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--arch_baseline_decay', type=float, default=0.99, help='weight decay for reward baseline')
parser.add_argument('--client', type=int, default=10, help='number of clients')
parser.add_argument('--glace_alpha', type=int, default=0, help='number of epoch for freezing alpha')
parser.add_argument('--glace_weight', type=int, default=1e6, help='number of epoch for freezing weight')
parser.add_argument('--warm_up', action='store_true', default=True, help='use trained model that has warmed up for 10k epochs')
parser.add_argument('--non_iid', action='store_true', default=False, help='use non iid dataset')
parser.add_argument('--fed_non_iid',action='store_true',default=False,help='use non iid distribution in FedNAS(CVPR20)')
parser.add_argument('--fed_selection',default=None,help='prepared distribution')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  criterion = nn.CrossEntropyLoss()
  global_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)

   
  weights_path='final/warmup_svhn.pt'
  utils.load(global_model, weights_path)

  # logging.info("param size = %fMB", utils.count_parameters_in_MB(global_model))

  global_optimizer = torch.optim.SGD(
      global_model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  #train_transform, valid_transform = _data_transforms_cifar10()
  #dataset = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
  '''
  testds=[]
  for (img, label) in dataset:
    if label in [0,2,3,5,9]:
      testds.append((img,label))
  random.shuffle(testds)
  dataset=copy.deepcopy(testds)
  '''
  
  train_queues = []

  fedselection=[[580,2259,44,999,265,119,1300,342,2448,0],
                [2,4167,2243,1164,0,0,0,0,0,0],
                [1663,60,86,2710,1,1749,330,781,0,0],
                [730,3429,1961,487,18,2471,0,0,0,0],
                [353,1011,432,79,2,53,1,1404,1745,3180],
                [945,538,3944,486,775,283,75,220,42,95],
                [228,313,153,226,3,194,628,60,451,0],
                [266,946,1133,1811,146,1458,3229,0,0,0],
                [3,1135,370,533,4685,192,18,310,0,0],
                [98,3,219,2,1563,363,146,2478,359,1384]]
  
  if args.fed_non_iid:
    train_queues=client_data(args.data,args.client,args.client_batch_size,fedselection)
  else:
    if args.non_iid:
      user_split = none_iid_split(dataset, num_user=args.client)
    else:
      user_split = even_split(dataset, args.client)

  
    for i in range(args.client):
      train_data = user_split[i]
      num_train = len(train_data)
      indices = list(range(num_train))
      train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.client_batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory=True, num_workers=2)

      train_queues.append(train_queue)

  # valid_data = user_split[-1]
  # num_train = len(valid_data)
  # indices = list(range(num_train))
  # valid_queue = torch.utils.data.DataLoader(
  #   valid_data, batch_size=32,
  #   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
  #   pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        global_optimizer, int(args.epochs), eta_min=args.learning_rate_min)
  global_architect = Architect(global_model, args)

  init_gradient(global_model)

  global_accuracy = []
  client_accuracy = []
  total_loss = []
      
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # genotype = global_model.genotype()
    # logging.info('genotype = %s', genotype)
    #


    client_models = []
    epoch_acc = []
    epoch_loss = []
    epoch_index_normal = []
    epoch_index_reduce = []
    for client_idx in range(args.client):
      mask_normal = sample_mask(global_model.alphas_normal)
      mask_reduce = sample_mask(global_model.alphas_reduce)
      index_normal = extract_index(mask_normal)
      index_reduce = extract_index(mask_reduce)
      client_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, criterion, mask_normal, mask_reduce)
      client_models.append(client_model)
      epoch_index_normal.append(index_normal)
      epoch_index_reduce.append(index_reduce)
    # copy weight of global model to client models
    # alphas in client models are actually gates, and equal to 1 forever
    client_weight_param(global_model, client_models)
    #parmsize=[utils.count_parameters_in_MB(x) for x in client_models]
    #logging.info(parmsize)
    #time_log=[]
    for client_idx in range(args.client):
      #torch.cuda.synchronize()
      #start=time.time()
      client_model = client_models[client_idx]
      client_models[client_idx], acc, loss = client_update(train_queues[client_idx], client_model, criterion)
      epoch_acc.append(acc)
      epoch_loss.append(loss)
      #torch.cuda.synchronize()
      #end=time.time()
      #times=end-start
      #time_log.append(times)
      
    #loggin.info("client running time:" +str(time_log))
    avg_acc = float(torch.mean(torch.Tensor(epoch_acc)))
    avg_loss = float(torch.mean(torch.Tensor(epoch_loss)))
    logging.info("client accuracy: " + str(epoch_acc))
    logging.info("client loss: " + str(epoch_loss))
    logging.info("client accuracy: "+str(avg_acc)+" , loss: "+str(avg_loss))
    client_accuracy.append(avg_acc)
    total_loss.append(avg_loss)

    fuse_weight_gradient(global_model,client_models)

    if epoch < args.glace_weight:
      global_optimizer.step()
      global_optimizer.zero_grad()

    if epoch > args.glace_alpha:
      global_architect.step(epoch_acc,epoch_index_normal,epoch_index_reduce)


    if (epoch+1) % args.report_freq == 0:
      # valid_acc, valid_obj = infer(valid_queue,global_model,criterion)
      # logging.info('valid_acc %f', valid_acc)
      # global_accuracy.append(valid_acc)
      logging.info("alphas normal")
      logging.info(F.softmax(global_model.alphas_normal, dim=-1))
      logging.info("alphas reduce")
      logging.info(F.softmax(global_model.alphas_reduce, dim=-1))
      logging.info("genotype")
      logging.info(global_model.genotype())
      utils.save(global_model, os.path.join(args.save, 'weights.pt'))

  logging.info("*** final log ***")
  logging.info("loss")
  logging.info(total_loss)
  logging.info("client accuracy")
  logging.info(client_accuracy)
  logging.info("global accuracy")
  logging.info(global_accuracy)


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model = model.cuda()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    # if step % args.report_freq == 0:
    #   logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  model = model.cpu()
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

