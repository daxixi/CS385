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
from noniid import client_data

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

  
  global_model.alphas_normal.data=torch.Tensor([[0.0023, 0.0033, 0.0030, 0.0031, 0.0038, 0.9782, 0.0032, 0.0031],
        [0.0049, 0.0058, 0.0051, 0.0059, 0.0060, 0.9614, 0.0055, 0.0055],
        [0.0275, 0.0279, 0.0282, 0.0249, 0.0248, 0.8187, 0.0253, 0.0229],
        [0.0964, 0.1418, 0.1256, 0.1482, 0.1132, 0.1810, 0.0860, 0.1078],
        [0.1020, 0.1736, 0.1807, 0.1727, 0.0876, 0.1060, 0.0956, 0.0819],
        [0.1088, 0.1323, 0.1247, 0.1334, 0.1415, 0.1433, 0.1149, 0.1012],
        [0.1168, 0.1147, 0.1422, 0.1427, 0.1272, 0.1259, 0.1225, 0.1081],
        [0.1042, 0.1565, 0.1257, 0.1553, 0.1189, 0.1259, 0.1050, 0.1084],
        [0.1417, 0.1257, 0.1066, 0.0983, 0.1485, 0.1118, 0.1287, 0.1388],
        [0.1140, 0.1101, 0.1483, 0.1510, 0.1086, 0.1315, 0.1301, 0.1063],
        [0.1173, 0.1395, 0.1226, 0.1046, 0.1259, 0.1110, 0.1446, 0.1344],
        [0.1084, 0.1408, 0.1255, 0.1424, 0.1193, 0.1214, 0.1219, 0.1203],
        [0.1441, 0.1196, 0.1340, 0.1015, 0.1309, 0.1130, 0.1418, 0.1151],
        [0.1427, 0.1385, 0.1234, 0.0913, 0.1265, 0.1323, 0.1079, 0.1374]])
  global_model.alphas_reduce=torch.Tensor([[0.0553, 0.1394, 0.1297, 0.1558, 0.1289, 0.1282, 0.1259, 0.1368],
        [0.0697, 0.1562, 0.1322, 0.1239, 0.1479, 0.1333, 0.1308, 0.1061],
        [0.0757, 0.1867, 0.1628, 0.1317, 0.1268, 0.1099, 0.1055, 0.1010],
        [0.0937, 0.1276, 0.1472, 0.1308, 0.1277, 0.1269, 0.1223, 0.1239],
        [0.1251, 0.1196, 0.1296, 0.1406, 0.1138, 0.1170, 0.1323, 0.1221],
        [0.1031, 0.1354, 0.1508, 0.1587, 0.1039, 0.1154, 0.1123, 0.1205],
        [0.0943, 0.1523, 0.1250, 0.1413, 0.1137, 0.1280, 0.1153, 0.1298],
        [0.1189, 0.1225, 0.1175, 0.1407, 0.1123, 0.1493, 0.1158, 0.1230],
        [0.1199, 0.1253, 0.1372, 0.1170, 0.1190, 0.1283, 0.1323, 0.1209],
        [0.1232, 0.1280, 0.1303, 0.1136, 0.1295, 0.1183, 0.1189, 0.1381],
        [0.1007, 0.1345, 0.1604, 0.1363, 0.1114, 0.0997, 0.1362, 0.1209],
        [0.1061, 0.1210, 0.1356, 0.1475, 0.1414, 0.1197, 0.1205, 0.1082],
        [0.1241, 0.1469, 0.1241, 0.1120, 0.1163, 0.1229, 0.1294, 0.1244],
        [0.1430, 0.1110, 0.1374, 0.1010, 0.1169, 0.1567, 0.1082, 0.1258]])
  
  '''
  if args.warm_up:
    logging.info("use warm up")
    if args.fed_non_iid:
      weights_path='final/warmup_noniid.pt'
    else:
      if args.client == 10:
        weights_path = 'final/warmup_weights.pt'
      elif args.client == 20:
        weights_path = 'final/warmup_weights_client20.pt'
      else:
        weights_path = 'final/warmup_weights_client50.pt'
      if args.non_iid:
        weights_path = 'final/warmup_weights_noniid.pt'
  '''

  weights_path='final/best_weight.pt'
  utils.load(global_model, weights_path)

  # logging.info("param size = %fMB", utils.count_parameters_in_MB(global_model))

  global_optimizer = torch.optim.SGD(
      global_model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  train_transform, valid_transform = _data_transforms_cifar10()
  dataset = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
  '''
  testds=[]
  for (img, label) in dataset:
    if label in [0,2,3,5,9]:
      testds.append((img,label))
  random.shuffle(testds)
  dataset=copy.deepcopy(testds)
  '''
  
  train_queues = []

  fedselection=[[20,166,805,1865,1297,1014,0,0,0,0],
                [318,1068,10,177,73,25,630,719,206,1249],
                [527,20,32,198,71,33,27,626,383,244],
                [271,1707,2,229,526,372,24,175,413,1788],
                [930,57,340,268,2215,2429,0,0,0,0],
                [226,83,2028,461,131,293,10,3134,0,0],
                [39,568,691,1584,370,669,1506,0,0,0],
                [1548,6,397,188,35,155,806,179,2216,0],
                [1107,22,294,17,0,264,1214,8,749,929],
                [14,3,401,13,18,10,783,159,1033,790]]
  
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

