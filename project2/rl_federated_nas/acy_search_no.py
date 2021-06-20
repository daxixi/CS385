import os
import sys
import time
import glob
import copy
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
from federated import sample_mask, client_update, fuse_weight_gradient, init_gradient, client_weight_param, extract_index,stale_generate
from data_distribution import _data_transforms_cifar10, even_split, none_iid_split
from stale import compute_stale_grad_weight

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--client_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
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
parser.add_argument('--glace_epoch', type=int, default=10000, help='number of epoch for freezing alpha')
parser.add_argument('--non_iid', action='store_true', default=False, help='use non iid dataset')
parser.add_argument('--stale',default=2,help='stale epochs')
parser.add_argument('--glace_alpha', type=int, default=0, help='number of epoch for freezing alpha')
parser.add_argument('--glace_weight', type=int, default=1e6, help='number of epoch for freezing weight')
parser.add_argument('--warm_up', action='store_true', default=True, help='use trained model that has warmed up for 10k epochs')
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
  # criterion = criterion.cuda()
  global_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  # global_model = global_model.cuda()
  # logging.info("param size = %fMB", utils.count_parameters_in_MB(global_model))
  if args.warm_up:
    logging.info("use warm up")
    if args.client == 10:
      weights_path = 'final/warmup_weights.pt'
    elif args.client == 20:
      weights_path = 'final/warmup_weights_client20.pt'
    else:
      weights_path = 'final/warmup_weights_client50.pt'
    if args.non_iid:
      weights_path = 'final/warmup_weights_noniid.pt'
    utils.load(global_model, weights_path)
    
  global_optimizer = torch.optim.SGD(
      global_model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  train_transform, valid_transform = _data_transforms_cifar10()
  dataset = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
  if args.non_iid:
    user_split = none_iid_split(dataset, num_user=args.client)
  else:
    user_split = even_split(dataset, args.client)

  train_queues = []
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

  #memorize something
  #Theta,A and G in last several epochs
  memory_weights=[]
  memory_alphas=[]
  memory_masks=[]
  #epoch i client j will finish and table[i][j] is its start epoch
  model_table = np.full((args.epochs + args.stale + 1, args.client), -(args.stale + 1))
  #to simulate also need to be recorded
  index_normal_old=[]
  gradient_old=[]
  index_reduce_old=[]
  acc_old=[]
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
    client_masks=[]
    for client_idx in range(args.client):
      mask_normal = sample_mask(global_model.alphas_normal)
      mask_reduce = sample_mask(global_model.alphas_reduce)
      client_masks.append([mask_normal,mask_reduce])
      index_normal = extract_index(mask_normal)
      index_reduce = extract_index(mask_reduce)
      client_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, criterion, mask_normal, mask_reduce)
      client_models.append(client_model)
      epoch_index_normal.append(index_normal)
      epoch_index_reduce.append(index_reduce)
    # copy weight of global model to client models
    # alphas in client models are actually gates, and equal to 1 forever
    client_weight_param(global_model, client_models)
    for client_idx in range(args.client):
      client_model = client_models[client_idx]
      client_models[client_idx], acc, loss = client_update(train_queues[client_idx], client_model, criterion)
      epoch_acc.append(acc)
      epoch_loss.append(loss)

    #simulate stale
    finish_epoch=stale_generate(args.client,args.stale)
    finish_epoch=[0 for _ in range(args.client)]
    for i in range(args.client):
      model_table[finish_epoch[i]+epoch][i]=epoch
    logging.info("stale epochs"+str(finish_epoch))

    avg_acc = float(torch.mean(torch.Tensor(epoch_acc)))
    avg_loss = float(torch.mean(torch.Tensor(epoch_loss)))
    logging.info("client accuracy: " + str(epoch_acc))
    logging.info("client loss: " + str(epoch_loss))
    logging.info("client accuracy: "+str(avg_acc)+" , loss: "+str(avg_loss))
    client_accuracy.append(avg_acc)
    total_loss.append(avg_loss)

    current_acc=[]
    current_index_normal=[]
    current_index_reduce=[]
    stale_alphas_normal=[]
    stale_alphas_reduce=[]
    stale_acc=[]
    stale_index_normal=[]
    stale_index_reduce=[]
    #prepare for soft syn
    #alphas
    for client_idx in range(args.client):
      if(epoch-model_table[epoch][client_idx]>args.stale):
        continue
      elif(epoch-model_table[epoch][client_idx]==0):
        current_acc.append(epoch_acc[client_idx])
        current_index_normal.append(epoch_index_normal[client_idx])
        current_index_reduce.append(epoch_index_reduce[client_idx])
        stale_index_normal.append(epoch_index_normal[client_idx])
        stale_index_reduce.append(epoch_index_reduce[client_idx])
        stale_acc.append(epoch_acc[client_idx])
        stale_alphas_normal.append(global_model.alphas_normal)
        stale_alphas_reduce.append(global_model.alphas_reduce)
      else:
        current_acc.append(epoch_acc[client_idx])
        current_index_normal.append(epoch_index_normal[client_idx])
        current_index_reduce.append(epoch_index_reduce[client_idx])
        stale_index_normal.append(index_normal_old[len(index_normal_old)-(epoch-model_table[epoch][client_idx])][client_idx])
        stale_index_reduce.append(index_reduce_old[len(index_reduce_old)-(epoch-model_table[epoch][client_idx])][client_idx])
        stale_acc.append(acc_old[len(acc_old)-(epoch-model_table[epoch][client_idx])][client_idx])
        stale_alphas_normal.append(memory_alphas[len(memory_alphas)-(epoch-model_table[epoch][client_idx])][0])
        stale_alphas_reduce.append(memory_alphas[len(memory_alphas)-(epoch-model_table[epoch][client_idx])][1])
    #weights
    use_models=[]
    for client_idx in range(args.client):
      if (epoch - model_table[epoch][client_idx] > args.stale):
        continue
      elif (epoch - model_table[epoch][client_idx] == 0):
        use_models.append(client_models[client_idx])
      else:
        old_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, criterion
                                  ,memory_masks[len(memory_masks)-(epoch-model_table[epoch][client_idx])][client_idx][0]
                                  ,memory_masks[len(memory_masks)-(epoch -model_table[epoch][client_idx])][client_idx][1])
        old_global_model=Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
        #parameters
        old_global_model.load_state_dict(memory_weights[len(memory_weights)-(epoch -model_table[epoch][client_idx])])
        client_weight_param(old_global_model,[old_model])
        #simulate old gradients
        new_iter=old_model.parameters()
        old_iter=gradient_old[len(gradient_old)-(epoch -model_table[epoch][client_idx])][client_idx]
        try:
          while True:
            old_weight = next(old_iter)
            new_weight = next(new_iter)
            new_weight.grad=old_weight.grad
        except StopIteration:
          pass
        
        new_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, criterion
                                ,memory_masks[len(memory_masks)-(epoch-model_table[epoch][client_idx])][client_idx][0]
                                ,memory_masks[len(memory_masks)-(epoch -model_table[epoch][client_idx])][client_idx][1])
        client_weight_param(global_model,[new_model])
        compute_stale_grad_weight(old_model,new_model)
        use_models.append(old_model)

    #refresh memorize things
    if (len(memory_weights)==args.stale+1):
      memory_weights=memory_weights[1:].copy()
      memory_weights.append(global_model.state_dict())
      memory_alphas=memory_alphas[1:].copy()
      memory_alphas.append([global_model.alphas_normal,global_model.alphas_reduce])
    else:
      memory_weights.append(global_model.state_dict())
      memory_alphas.append([global_model.alphas_normal,global_model.alphas_reduce])
    if(len(memory_masks)==args.stale):
      memory_masks=memory_masks[1:].copy()
      memory_masks.append(client_masks)
    else:
      memory_masks.append(client_masks)
    gradients=[]
    for client_idx in range(args.client):
      gradients.append(client_models[client_idx].parameters())
    if(len(acc_old)==args.stale):
      acc_old=acc_old[1:].copy()
      acc_old.append(epoch_acc)
      index_normal_old=index_normal_old[1:].copy()
      index_normal_old.append(epoch_index_normal)
      index_reduce_old=index_reduce_old[1:].copy()
      index_reduce_old.append(epoch_index_reduce)
      gradient_old=gradient_old[1:].copy()
      gradient_old.append(gradients)
    else:
      acc_old.append(epoch_acc)
      index_normal_old.append(epoch_index_normal)
      index_reduce_old.append(epoch_index_reduce)
      gradient_old.append(gradients)

    if(len(use_models))>0:
      fuse_weight_gradient(global_model,use_models)

    if epoch < args.glace_weight:
      global_optimizer.step()
      global_optimizer.zero_grad()

    if epoch > args.glace_alpha:
      if(len(stale_acc)>0):
        global_architect.stale_step(current_acc, current_index_normal, current_index_reduce,
                                  stale_alphas_normal, stale_alphas_reduce, stale_acc, stale_index_normal, stale_index_reduce)

    # if epoch > args.glace_epoch:
    #   global_architect.step(epoch_acc,epoch_index_normal,epoch_index_reduce)

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
      utils.save(global_model, os.path.join(args.save, 'weights_epoch'+str(epoch)+'.pt'))

  logging.info("*** final log ***")
  logging.info("loss")
  logging.info(total_loss)
  logging.info("client accuracy")
  logging.info(client_accuracy)
  logging.info("global accuracy")
  logging.info(global_accuracy)
  utils.save(global_model, os.path.join(args.save, 'finalweights.pt'))

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model = model.cuda()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

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

