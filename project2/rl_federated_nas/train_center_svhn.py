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
parser.add_argument('--arch', type=str, default='svhn', help='which architecture to use')
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

    genotype = eval("genotypes.%s" % args.arch)
    global_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
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
    train_data = dset.SVHN(root=args.data, split='train',download=True, transform=train_transform)
    valid_data = dset.SVHN(root=args.data,split='test',download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, float(args.epochs))
    scheduler =torch.optim.lr_scheduler.MultiStepLR(global_optimizer, [10,20,30], gamma=0.1, last_epoch=-1)#Densenet

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        global_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        global_model, train_acc, train_obj = train(train_queue, global_model, criterion, global_optimizer)
        logging.info('train_acc %f', train_acc)

        if (epoch+1)%1==0:
            valid_acc, valid_obj = infer(valid_queue, global_model, criterion)
            logging.info('valid_acc %f', valid_acc)

            utils.save(global_model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target,volatile=False).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return model,top1.avg, objs.avg

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

            logits, _ = model(input)
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

