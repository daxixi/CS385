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
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

from torch.autograd import Variable
from model_search import Network
from model_search_local import MaskedNetwork
from architect import Architect
from federated import sample_mask, client_update, fuse_weight_gradient, init_gradient, client_weight_param, extract_index,stale_generate
from data_distribution import _data_transforms_cifar10, even_split, none_iid_split
from stale import compute_stale_grad_weight

from torchvision.models import vgg11

### please use pytorch 1.5.1 virtual environment ###

parser = argparse.ArgumentParser(
    description="Federated NAS",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
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
parser.add_argument('--glace_alpha', type=int, default=0, help='number of epoch for freezing alpha')
parser.add_argument('--glace_weight', type=int, default=1e6, help='number of epoch for freezing weight')
parser.add_argument('--warm_up', action='store_true', default=True,
                    help='use trained model that has warmed up for 10k epochs')
parser.add_argument('--non_iid', action='store_true', default=False, help='use non iid dataset')
parser.add_argument('--stale',default=2,help='stale epochs')

parser.add_argument('--devices', default='[2,3]', type=str, help='List of visible devices')

args = parser.parse_args()

CIFAR_CLASSES = 10

# list of available GPU devices
device_list = eval(args.devices)
world_size = len(device_list)
task_per_worker = args.client // world_size
# assign num of task for each worker
num_tasks = [task_per_worker for _ in range(world_size)]
num_tasks[-1] += (args.client % world_size)
# num of workers + num of server(1)
world_size += 1


# remote hook functions
def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class Worker:
    def __init__(self, device_id, num_task):
        self.id = rpc.get_worker_info().id
        self.device_id = device_id
        torch.cuda.set_device(device_id)
        self.num_task = num_task  # one worker compute multiple tasks
        print("Worker id", self.id, "device id", device_id, "num_task", self.num_task)
        self.criterion = nn.CrossEntropyLoss()

    # replace param with gradient for transfer
    def _wrap_gradient(self, model):
        iter = model.parameters()
        try:
            while True:
                param = next(iter)
                param.data = param.grad
        except StopIteration:
            pass

    def recieve_queues(self, agent_rref):
        train_queues = _remote_method(Server.send_queues, agent_rref, self.id, self.num_task)
        self.train_queues = copy.deepcopy(train_queues)
        # print(self.id," train_queues_recived")
        # print(train_queues)

    def run_episode(self, agent_rref):
        torch.cuda.set_device(self.device_id)
        for i in range(self.num_task):
            task_idx = (self.id - 1) * task_per_worker + i
            # receive sub-model from server
            # print(self.id,":",i,"ready to recive")
            model = _remote_method(Server.send_model, agent_rref, task_idx)
            model = copy.deepcopy(model)
            # print("model recieved")
            # train sub-model with local data
            model, acc, loss = client_update(self.train_queues[i], model, self.criterion)
            # wrap gradient into param
            self._wrap_gradient(model)
            # send gradient to server
            # besides, send model accuracy to server (ignored here)
            # print(self.id, ":", i, "ready to return")
            _remote_method(Server.collect_result, agent_rref, task_idx, model, acc, loss)


class Server:
    def __init__(self, world_size):
        args.save = 'dis-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

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

        self.train_queues = []
        train_transform, valid_transform = _data_transforms_cifar10()
        dataset = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        # even split dataset
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
                pin_memory=True, num_workers=0)

            self.train_queues.append(train_queue)

        self.criterion = nn.CrossEntropyLoss()
        self.global_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, self.criterion)
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
            utils.load(self.global_model, weights_path)

        # logging.info("param size = %fMB", utils.count_parameters_in_MB(global_model))

        self.global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.global_optimizer, int(args.epochs), eta_min=args.learning_rate_min)
        self.global_architect = Architect(self.global_model, args)

        init_gradient(self.global_model)

        self.global_accuracy = []
        self.client_accuracy = []
        self.total_loss = []

        # memorize something
        # Theta,A and G in last several epochs
        self.memory_weights = []
        self.memory_alphas = []
        self.memory_masks = []
        # epoch i client j will finish and table[i][j] is its start epoch
        self.model_table = np.full((args.epochs + args.stale + 1, args.client), -(args.stale + 1))
        # to simulate also need to be recorded
        self.index_normal_old = []
        self.gradient_old = []
        self.index_reduce_old = []
        self.acc_old = []

        self.wk_rrefs = []
        self.agent_rref = RRef(self)
        # construct workers
        for wk_rank in range(1, world_size):
            wk_info = rpc.get_worker_info(WORKER_NAME.format(wk_rank))
            self.wk_rrefs.append(
                remote(
                    wk_info,
                    Worker,
                    args=(device_list[wk_rank - 1], num_tasks[wk_rank - 1])
                )
            )
        # container of worker results
        self.rewards = [[] for _ in range(args.client)]
        # prepare data for each worker
        futs = []
        for wk_rref in self.wk_rrefs:
            futs.append(
                rpc_async(
                    wk_rref.owner(),
                    _call_method,
                    args=(Worker.recieve_queues, wk_rref, self.agent_rref)
                )
            )
        # time.sleep(200)
        for fut in futs:
            fut.wait()

    def send_queues(self, id, num):
        queues = copy.deepcopy(self.train_queues[(id - 1) * task_per_worker:(id - 1) * task_per_worker + num])
        return queues

    # send sub-models to workers
    def send_model(self, task_idx):
        model = self.client_models[task_idx]
        model = copy.deepcopy(model)
        return model

    # collect results from workers
    def collect_result(self, task_idx, model, acc, loss):
        self.epoch_acc[task_idx] = acc
        self.epoch_loss[task_idx] = loss
        self.client_models[task_idx] = copy.deepcopy(model)
        # print(task_idx)
        # print(self.rewards[task_idx])

    # update super-net with workers' results
    def server_update(self, epoch):
        # simulate stale
        finish_epoch = stale_generate(args.client, args.stale)
        for i in range(args.client):
            self.model_table[finish_epoch[i] + epoch][i] = epoch
        logging.info("stale epochs" + str(finish_epoch))

        avg_acc = float(torch.mean(torch.Tensor(self.epoch_acc)))
        avg_loss = float(torch.mean(torch.Tensor(self.epoch_loss)))
        logging.info("client accuracy: " + str(self.epoch_acc))
        logging.info("client loss: " + str(self.epoch_loss))
        logging.info("client accuracy: " + str(avg_acc) + " , loss: " + str(avg_loss))
        self.client_accuracy.append(avg_acc)
        self.total_loss.append(avg_loss)

        for model in self.client_models:
            iter = model.parameters()
            try:
                while True:
                    param = next(iter)
                    param.grad = param.data
            except StopIteration:
                pass

        current_acc = []
        current_index_normal = []
        current_index_reduce = []
        stale_alphas_normal = []
        stale_alphas_reduce = []
        stale_acc = []
        stale_index_normal = []
        stale_index_reduce = []
        # prepare for soft syn
        # alphas
        for client_idx in range(args.client):
            if (epoch - self.model_table[epoch][client_idx] > args.stale):
                continue
            elif (epoch - self.model_table[epoch][client_idx] == 0):
                current_acc.append(self.epoch_acc[client_idx])
                current_index_normal.append(self.epoch_index_normal[client_idx])
                current_index_reduce.append(self.epoch_index_reduce[client_idx])
                stale_index_normal.append(self.epoch_index_normal[client_idx])
                stale_index_reduce.append(self.epoch_index_reduce[client_idx])
                stale_acc.append(self.epoch_acc[client_idx])
                stale_alphas_normal.append(self.global_model.alphas_normal)
                stale_alphas_reduce.append(self.global_model.alphas_reduce)
            else:
                current_acc.append(self.epoch_acc[client_idx])
                current_index_normal.append(self.epoch_index_normal[client_idx])
                current_index_reduce.append(self.epoch_index_reduce[client_idx])
                stale_index_normal.append(
                    self.index_normal_old[len(self.index_normal_old) - (epoch - self.model_table[epoch][client_idx])][client_idx])
                stale_index_reduce.append(
                    self.index_reduce_old[len(self.index_reduce_old) - (epoch - self.model_table[epoch][client_idx])][client_idx])
                stale_acc.append(self.acc_old[len(self.acc_old) - (epoch - self.model_table[epoch][client_idx])][client_idx])
                stale_alphas_normal.append(
                    self.memory_alphas[len(self.memory_alphas) - (epoch - self.model_table[epoch][client_idx])][0])
                stale_alphas_reduce.append(
                    self.memory_alphas[len(self.memory_alphas) - (epoch - self.model_table[epoch][client_idx])][1])
        # weights
        use_models = []
        for client_idx in range(args.client):
            if (epoch - self.model_table[epoch][client_idx] > args.stale):
                continue
            elif (epoch - self.model_table[epoch][client_idx] == 0):
                use_models.append(self.client_models[client_idx])
            else:
                old_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, self.criterion
                                          , self.memory_masks[len(self.memory_masks) - (epoch - self.model_table[epoch][client_idx])][
                                              client_idx][0]
                                          , self.memory_masks[len(self.memory_masks) - (epoch - self.model_table[epoch][client_idx])][
                                              client_idx][1])
                old_global_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, self.criterion)
                # parameters
                old_global_model.load_state_dict(
                    self.memory_weights[len(self.memory_weights) - (epoch - self.model_table[epoch][client_idx])])
                client_weight_param(old_global_model, [old_model])
                # simulate old gradients
                new_iter = old_model.parameters()
                old_iter = self.gradient_old[len(self.gradient_old) - (epoch - self.model_table[epoch][client_idx])][client_idx]
                try:
                    while True:
                        old_weight = next(old_iter)
                        new_weight = next(new_iter)
                        new_weight.grad = old_weight.grad
                except StopIteration:
                    pass

                new_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, self.criterion
                                          , self.memory_masks[len(self.memory_masks) - (epoch - self.model_table[epoch][client_idx])][
                                              client_idx][0]
                                          , self.memory_masks[len(self.memory_masks) - (epoch - self.model_table[epoch][client_idx])][
                                              client_idx][1])
                client_weight_param(self.global_model, [new_model])
                compute_stale_grad_weight(old_model, new_model)
                use_models.append(old_model)

        # refresh memorize things
        if (len(self.memory_weights) == args.stale + 1):
            self.memory_weights = self.memory_weights[1:].copy()
            self.memory_weights.append(self.global_model.state_dict())
            self.memory_alphas = self.memory_alphas[1:].copy()
            self.memory_alphas.append([self.global_model.alphas_normal, self.global_model.alphas_reduce])
        else:
            self.memory_weights.append(self.global_model.state_dict())
            self.memory_alphas.append([self.global_model.alphas_normal, self.global_model.alphas_reduce])
        if (len(self.memory_masks) == args.stale):
            self.memory_masks = self.memory_masks[1:].copy()
            self.memory_masks.append(self.client_masks)
        else:
            self.memory_masks.append(self.client_masks)
        gradients = []
        for client_idx in range(args.client):
            gradients.append(self.client_models[client_idx].parameters())
        if (len(self.acc_old) == args.stale):
            self.acc_old = self.acc_old[1:].copy()
            self.acc_old.append(self.epoch_acc)
            self.index_normal_old = self.index_normal_old[1:].copy()
            self.index_normal_old.append(self.epoch_index_normal)
            self.index_reduce_old = self.index_reduce_old[1:].copy()
            self.index_reduce_old.append(self.epoch_index_reduce)
            self.gradient_old = self.gradient_old[1:].copy()
            self.gradient_old.append(gradients)
        else:
            self.acc_old.append(self.epoch_acc)
            self.index_normal_old.append(self.epoch_index_normal)
            self.index_reduce_old.append(self.epoch_index_reduce)
            self.gradient_old.append(gradients)

        if (len(use_models)) > 0:
            fuse_weight_gradient(self.global_model, use_models)

        if epoch < args.glace_weight:
            self.global_optimizer.step()
            self.global_optimizer.zero_grad()

        if epoch > args.glace_alpha:
            if (len(stale_acc) > 0):
                self.global_architect.stale_step(current_acc, current_index_normal, current_index_reduce,
                                            stale_alphas_normal, stale_alphas_reduce, stale_acc, stale_index_normal,
                                            stale_index_reduce)

        if (epoch + 1) % args.report_freq == 0:
            # valid_acc, valid_obj = infer(valid_queue,global_model,criterion)
            # logging.info('valid_acc %f', valid_acc)
            # global_accuracy.append(valid_acc)
            logging.info("alphas normal")
            logging.info(F.softmax(self.global_model.alphas_normal, dim=-1))
            logging.info("alphas reduce")
            logging.info(F.softmax(self.global_model.alphas_reduce, dim=-1))
            logging.info("genotype")
            logging.info(self.global_model.genotype())
            utils.save(self.global_model, os.path.join(args.save, 'weights_epoch' + str(epoch) + '.pt'))

    # instruct workers to start an episode
    def run_episode(self, episode):
        self.scheduler.step()
        lr = self.scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', episode, lr)

        self.client_models = []
        self.epoch_acc = [[] for _ in range(args.client)]
        self.epoch_loss = [[] for _ in range(args.client)]
        self.epoch_index_normal = []
        self.epoch_index_reduce = []
        self.client_masks = []

        for client_idx in range(args.client):
            mask_normal = sample_mask(self.global_model.alphas_normal)
            mask_reduce = sample_mask(self.global_model.alphas_reduce)
            self.client_masks.append([mask_normal, mask_reduce])
            index_normal = extract_index(mask_normal)
            index_reduce = extract_index(mask_reduce)
            client_model = MaskedNetwork(args.init_channels, CIFAR_CLASSES, args.layers, self.criterion, mask_normal,
                                         mask_reduce)
            self.client_models.append(client_model)
            self.epoch_index_normal.append(index_normal)
            self.epoch_index_reduce.append(index_reduce)
        client_weight_param(self.global_model, self.client_models)

        # self.client_models=[vgg11() for _ in range(args.client)]

        futs = []
        for wk_rref in self.wk_rrefs:
            futs.append(
                rpc_async(
                    wk_rref.owner(),
                    _call_method,
                    args=(Worker.run_episode, wk_rref, self.agent_rref)
                )
            )
        # time.sleep(200)
        for fut in futs:
            fut.wait()
        self.server_update(episode)


SERVER_NAME = "agent"
WORKER_NAME = "obs{}"


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the server
        rpc.init_rpc(SERVER_NAME, rank=rank, world_size=world_size)
        server = Server(world_size)
        for episode in range(args.epochs):
            server.run_episode(episode)
        logging.info("*** final log ***")
        logging.info("loss")
        logging.info(server.total_loss)
        logging.info("client accuracy")
        logging.info(server.client_accuracy)
        logging.info("global accuracy")
        logging.info(server.global_accuracy)
    else:
        # other ranks are workers
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size)
        # workers passively waiting for instructions from the server
    rpc.shutdown()


if __name__ == '__main__':
    # process group for server and workers
    mp.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )





