import os
import numpy as np
import time
import argparse

import sys

from mpi4py import MPI
from math import ceil
import random
import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
cudnn.benchmark = True

import models.resnet
import models.vggnet
import models.wrn
import util
from graph_manager import FixedProcessor, MatchaProcessor
from communicator import decenCommunicator, ChocoCommunicator, centralizedCommunicator

def sync_allreduce(model, rank, size):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()  # PyTorch 张量的模型参数数据复制到 NumPy 数组中
        recvdata[param] = np.empty(senddata[param].shape, dtype = senddata[param].dtype)  # 创建了一个具有相同形状和数据类型的空 NumPy 数组
    # torch.cuda.synchronize() # 用于确保 GPU 操作已完成
    comm.barrier() # 用于确保所有进程在继续之前都已完成。

    comm_start = time.time()
    for param in model.parameters(): # 对每个模型参数进行总和聚合。
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    # torch.cuda.synchronize()
    comm.barrier()
    
    comm_end = time.time()
    comm_t = (comm_end - comm_start)
        
    for param in model.parameters():
        param.data = torch.Tensor(recvdata[param])# .cuda()
        param.data = param.data/float(size)    # 子图不同，size也不同？？？
    return comm_t

def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed+rank)
    np.random.seed(args.randomSeed)

    # load data
    train_loader, test_loader = util.partition_dataset(rank, size, args)    
    num_batches = ceil(len(train_loader.dataset) / float(args.bs))

    # 负荷基础网络拓扑结构
    subGraphs = util.select_graph(args.graphid)
    
    # 定义图形激活方案
    if args.matcha:
        GP = MatchaProcessor(subGraphs, args.budget, rank, size, args.epoch*num_batches, False)
    else:
        GP = FixedProcessor(subGraphs, args.budget, rank, size, args.epoch*num_batches, True)

    # define communicator
    if args.compress:
        communicator = ChocoCommunicator(rank, size, GP, 0.9, args.consensus_lr)
    else:
        communicator = decenCommunicator(rank, size, GP)

    # select neural network model
    model = util.select_model(100, args)
    # model = model.cuda()
    criterion = nn.CrossEntropyLoss()# .cuda()
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=5e-4,
                          nesterov=args.nesterov,
                          dampening=0)
    # 这是一个布尔值，用于指定是否使用 Nesterov 动量。Nesterov 动量是一种改进的动量算法，可以更好地处理梯度下降中的震荡问题。
    
    # 保证所有本地模型都从同一点出发
    # can be removed    
    sync_allreduce(model, rank, size)

    # init recorder
    comp_time, comm_time = 0, 0
    recorder = util.Recorder(args, rank)
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    tic = time.time()
    itr = 0


    # start training
    for epoch in range(args.epoch):
        model.train()

        # Start training each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()
            # data loading 
            # data, target = data.cuda(non_blocking = True), target.cuda(non_blocking = True)

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            # record training loss and accuracy
            record_start = time.time()
            acc1 = util.comp_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            record_end = time.time()

            # backward pass
            loss.backward()
            update_learning_rate(optimizer, epoch, itr=batch_idx, itr_per_epoch=len(train_loader))

            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()

            d_comp_time = (end_time - start_time - (record_end - record_start))
            comp_time += d_comp_time

            # 交流在这里发生
            d_comm_time = communicator.communicate(model)
            comm_time += d_comm_time

            print("batch_idx: %d, rank: %d, comp_time: %.3f, comm_time: %.3f,epoch time: %.3f " % (batch_idx+1,rank,d_comp_time, d_comm_time, comp_time+ comm_time), end='\r')

        toc = time.time()
        record_time = toc - tic # time that includes anything
        epoch_time = comp_time + comm_time # only include important parts

        # evaluate test accuracy at the end of each epoch
        test_acc = util.test(model, test_loader)

        recorder.add_new(record_time,comp_time,comm_time,epoch_time,top1.avg,losses.avg,test_acc)
        print(f"{epoch:0>3}  rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, test_acc: %.3f epoch time: %.3f" % (rank, epoch, losses.avg, top1.avg, test_acc, epoch_time))
        if rank == 0:
            print("---comp_time: %.3f, comm_time: %.3f, comp_time_budget: %.3f, comm_time_budget: %.3f" % (comp_time, comm_time, comp_time/epoch_time, comm_time/epoch_time))
       
        if epoch%10 == 0:
            recorder.save_to_file()

        # reset recorders
        comp_time, comm_time = 0, 0
        losses.reset()
        top1.reset()
        tic = time.time()

    recorder.save_to_file()


def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    target_lr = args.lr
    lr_schedule = [100, 150]

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= base_lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    else:
        lr = target_lr
        for e in lr_schedule:
            if epoch >= e:
                lr *= 0.1

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--name', default="Vanilla DecenSGD-cifar100", type=str, help='experiment name')                   # gai
    parser.add_argument('--description', default="Vanilla DecenSGD-cifar100", type=str, help='experiment description')     # gai

    parser.add_argument('--model', default="wrn", type=str, help='model name: res/VGG/wrn')                 # gai
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=200, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--warmup', default=True, action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', default=False, action='store_true', help='use nesterov momentum or not')

    parser.add_argument('--matcha', default=True, action='store_true', help='use MATCHA or not')            # gai
    parser.add_argument('--budget', default=1, type=float, help='comm budget')                            # gai
    parser.add_argument('--graphid', default=0, type=int, help='the idx of base graph')                     # gai
    
    parser.add_argument('--dataset', default='cifar100', type=str, help='the dataset')                       # gai
    parser.add_argument('--datasetRoot', default='./dataset', type=str, help='the path of dataset')
    parser.add_argument('--p', '-p', default=True, action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath', default='./saveModel', type=str, help='save path')
    parser.add_argument('--save', default=True, action='store_true', help='save medal or not')
    
    parser.add_argument('--compress', default=False, action='store_true', help='use chocoSGD or not')        # gai
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--randomSeed', default=1234, type=int, help='random seed')

    args = parser.parse_args()

    if not args.description:
        print('No experiment description, exit!')
        exit()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    run(rank, size)
    # mpiexec -n 8 -machinefile c8 python train_mpi.py
    # mpiexec -n 8  python train_mpi.py
