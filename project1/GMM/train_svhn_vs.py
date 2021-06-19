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

from torch.autograd import Variable
#from model import NetworkCIFAR as Network
import GM
import MobileNet
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
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

  model = MobileNet.get_model()
  utils.load(model,'weights.pt')
  model = model.cuda()
  GMmodel = GM.GMM(1024)
  print(GMmodel)
  GMmodel = GMmodel.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

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
  ct=0
  for input,target in train_queue:
    if ct>40:
      break
    ct+=1
    input = Variable(input).cuda()
    _,input = model(input)
    input=input.detach().cpu()
    for i in range(input.shape[0]):
      train_dataset.append(input[i].numpy().tolist())
      train_labels.append(target[i])
  train_dataset=np.array(train_dataset)
  train_labels=np.array(train_labels)
  ct=0
  for input,target in valid_queue:
    if ct>40:
      break
    ct+=1
    input = Variable(input).cuda()
    _,input = model(input)
    input=input.detach().cpu()
    for i in range(input.shape[0]):
      valid_dataset.append(input[i].numpy().tolist())
      valid_labels.append(target[i])
  valid_dataset=np.array(valid_dataset)
  valid_labels=np.array(valid_labels)

  train_dataset=torch.from_numpy(train_dataset).unsqueeze(1).cuda()
  GMmodel.fit(train_dataset)
  train_predict=GMmodel.predict(train_dataset)
  valid_dataset=torch.from_numpy(valid_dataset).unsqueeze(1).cuda()
  valid_predict=GMmodel.predict(valid_dataset)
  #use first 100 pics for voter
  voters=np.zeros((10,10))
  for p,t in zip(train_predict,train_labels):
    voters[p][t]+=1
  project=np.zeros(10)
  for i in range(10):
    project[i]=np.argmax(voters[i])
  project=torch.from_numpy(project)
  for i in range(train_predict.shape[0]):
    train_predict[i]=project[valid_predict[i]]
  for i in range(valid_predict.shape[0]):
    valid_predict[i]=project[valid_predict[i]]
  acc=accuracy_score(train_labels,train_predict)
  logging.info(acc)
  acc=accuracy_score(valid_labels,valid_predict)
  logging.info(acc)
  
    
if __name__ == '__main__':
  main() 

