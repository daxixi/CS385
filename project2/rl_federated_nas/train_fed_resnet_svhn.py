import os
import sys
import time
import glob
import copy
import numpy as np
import torch
import random
import copy
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.models as models

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from data_distribution import _data_transforms_cifar10, even_split, none_iid_split
from noniid_svhn import client_data

parser = argparse.ArgumentParser("svhn")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--client_batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=6000, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--client', type=int, default=10, help='number of clients')
parser.add_argument('--non_iid', action='store_true', default=False, help='use non iid dataset')
parser.add_argument('--fed_non_iid',action='store_true',default=True, help='use non iid distribution in FedNAS(CVPR20)')
parser.add_argument('--fed_selection',default=None,help='prepared distribution')
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

global_accuracy = []
client_accuracy = []
total_loss = []

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    global_model = models.resnet152(num_classes=CIFAR_CLASSES)
    global_model = global_model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(global_model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    global_optimizer = torch.optim.SGD(
        global_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.SVHN(root=args.data,split='train', download=True, transform=train_transform)
    valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

    _train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    '''
    testds=[]
    for (img, label) in valid_data:
        if label in [0,2,3,5,9]:
          testds.append((img,label))
    random.shuffle(testds)
    valid_data=copy.deepcopy(testds)
    '''

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    
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
    
    train_queues = []
    if args.fed_non_iid:
        train_queues = client_data(args.data, args.client, args.client_batch_size,fedselection)
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, float(args.epochs))

    init_gradient(global_model,_train_queue,criterion)

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        global_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        global_model, train_acc, train_obj = train(train_queues, global_model, criterion, global_optimizer)
        logging.info('train_acc %f', train_acc)

        if (epoch+1)%100==0:
            valid_acc, valid_obj = infer(valid_queue, global_model, criterion)
            logging.info('valid_acc %f', valid_acc)

            utils.save(global_model, os.path.join(args.save, 'weights.pt'))


def train(train_queues, model, criterion, optimizer):
    client_models = []
    epoch_acc = []
    epoch_loss = []
    
    optimizer.zero_grad()
    model=model.cpu()

    for client_idx in range(args.client):
      client_models.append(copy.deepcopy(model))
      client_models[client_idx], acc, loss = client_update(train_queues[client_idx], client_models[client_idx], criterion, client_idx)
      epoch_acc.append(acc)
      epoch_loss.append(loss)

    avg_acc = float(torch.mean(torch.Tensor(epoch_acc)))
    avg_loss = float(torch.mean(torch.Tensor(epoch_loss)))
    logging.info("client accuracy: " + str(epoch_acc))
    logging.info("client loss: " + str(epoch_loss))
    logging.info("client accuracy: "+str(avg_acc)+" , loss: "+str(avg_loss))
    client_accuracy.append(avg_acc)
    total_loss.append(avg_loss)

    model=fuse_weight_gradient(model, client_models)

    optimizer.step()
    model=model.cuda()

    return model,avg_acc,avg_loss

def init_gradient(model,train_queue,criterion):
    logging.info("initializing gradient")
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda()
        logits= model(input)
        loss = criterion(logits, target)
        #if args.auxiliary:
        #    loss_aux = criterion(logits_aux, target)
        #    loss += args.auxiliary_weight * loss_aux
        loss.backward()
        break
    model.zero_grad()

def fuse_weight_gradient(model,client_models):
    global_iter=model.parameters()
    client_iters=[x.parameters() for x in client_models]
    try:
        while True:
            global_weight=next(global_iter)
            for citer in client_iters:
                client_weight=next(citer)
                global_weight.grad=global_weight.grad+client_weight.grad
            global_weight.grad=global_weight.grad/len(client_models)
    except StopIteration:
        pass
    return model

def client_update(train_queue, model, criterion, num):
    criterion = criterion.cuda()
    model = model.cuda()
    model.train()
    prec1 = 0
    loss = 0
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda()
        logits= model(input)
        loss = criterion(logits, target)
        #if args.auxiliary:
        #    loss_aux = criterion(logits_aux, target)
        #    loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        break

    model = model.cpu()

    return model, prec1.cpu().item(), loss.cpu().item()

def infer(valid_queue, model, criterion):
    criterion=criterion.cuda()
    model =model.cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    prec1=0
    prec5=0
    loss=0
    model.train()

    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda()

            logits= model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.cpu().item(), n)
        top1.update(prec1.cpu().item(), n)
        top5.update(prec5.cpu().item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

