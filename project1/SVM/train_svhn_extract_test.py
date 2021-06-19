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
import MobileNet

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sklearn.externals
import pickle

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
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

CIFAR_CLASSES = 10


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

  model = MobileNet.get_model()
  utils.load(model,'weights.pt')
  model = model.cuda()
  train_dataset=[]
  train_labels=[]
  valid_dataset=[]
  valid_labels=[]
  
  for input,target in valid_queue:
    input = Variable(input).cuda()
    _,input = model(input)
    input=input.detach().cpu()
    input=input.view(input.size(0),-1)
    for i in range(input.shape[0]):
      valid_dataset.append(input[i].numpy().tolist())
      valid_labels.append(target[i])
  valid_dataset=np.array(valid_dataset)
  valid_labels=np.array(valid_labels)


  with open('eval-linearext-20210602-205849/model.pickle','rb') as f:
    clf=pickle.load(f)
  logits=clf.predict(valid_dataset)
  acc=accuracy_score(valid_labels,logits)
  print(acc)


if __name__ == '__main__':
  main() 

