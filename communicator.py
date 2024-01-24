import numpy as np
import time
import torch
from mpi4py import MPI
from compressors import get_top_k

from comm_helpers import flatten_tensors, unflatten_tensors


class Communicator(object):
    """ Classs designed for communicating local models at workers
     分布式机器学习环境中进行通信和模型参数同步"""
    def __init__(self, rank, size):
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size

    def communicate(self, model):
        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocessing
        self.prepare_comm_buffer()

        # communication happens here
        # record the communication time
        comm_time = self.averaging()

        # Update local models
        self.reset_model()

        return comm_time

    def prepare_comm_buffer(self):
        raise NotImplemented

    def averaging(self):
        raise NotImplemented

    def reset_model(self):
        raise NotImplemented


class centralizedCommunicator(Communicator):
    """ Perform AllReduce at each iteration """
    def __init__(self, rank, size):
        super(centralizedCommunicator, self).__init__(rank, size)

    
    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()

    def averaging(self):
        self.comm.barrier() # 所有进程在同一时刻等待，直到所有进程都到达了这一行代码，然后它们才能继续执行后续的操作。
        tic = time.time()

        # AllReduce
        self.recv_buffer = self.comm.allreduce(self.send_buffer, op=MPI.SUM)  # 将所有进程的模型参数进行加和操作，以获得全局平均。
        self.recv_buffer.div_(self.size)
        
        self.comm.barrier()
        toc = time.time()

        return toc - tic

    def reset_model(self):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors( # 将一个或多个可迭代对象（例如列表、元组、字符串等）按照相同索引的元素组合成元组对
                            self.recv_buffer,# .cuda(),
                            self.tensor_list),
                            self.tensor_list):
                t.copy_(f)  # 赋值


class decenCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(decenCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)

    def averaging(self, active_flags):
        
        self.comm.barrier()
        tic = time.time()

        # 分散平均
        degree = 0 # 记录每个节点的度数
        for graph_id, flag in enumerate(active_flags): # 同时访问元素和其索引
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]

                    # 实现进程之间的双向数据交换，其中每个进程同时向其邻居发送数据并接收来自邻居的数据。
                    # source=neighbor_rank：这指定了当前进程期望从中接收数据的源进程。
                    # dest=neighbor_rank：这指定了当前进程希望将数据发送到的目标进程。
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest=neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer.add_(self.recv_tmp, alpha=self.neighbor_weight)
        
        # 根据权重计算度
        selfweight = 1 - degree * self.neighbor_weight
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer.add_(self.send_buffer, alpha=selfweight)

        self.comm.barrier()
        toc = time.time()

        return toc - tic

    def reset_model(self):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.recv_buffer,# .cuda(),
                            self.tensor_list),
                            self.tensor_list):
                t.copy_(f)

    def communicate(self, model):
        # 获取当前迭代的激活拓扑
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # 如果没有子图被激活、
        # 然后直接开始下一次迭代
        if np.sum(active_flags) == 0:
            return 0

        # 将所有模型参数堆叠成一个张量列表
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # 必要的预处理
        self.prepare_comm_buffer()

        # 根据激活的拓扑结构进行分散平均
        # 记录通信时间
        comm_time = self.averaging(active_flags)

        # 更新本地模型
        self.reset_model()

        return comm_time


class ChocoCommunicator(Communicator):
    """ decentralized averaging using compressed gradients (top-k) """
    
    def __init__(self, rank, size, topology, ratio, consensus_lr):
        super(ChocoCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

        self.initialized = False
        self.consensus_lr = consensus_lr
        self.ratio = ratio


    def prepare_comm_buffer(self):
        # flatten tensors
        # If not initialized, then initialize x_hat and s
        self.x = flatten_tensors(self.tensor_list).cpu()
        if not self.initialized:
            self.x_hat = torch.zeros_like(self.x)
            self.s = torch.zeros_like(self.x)
            self.initialized = True

        tic = time.time()
        # get compressed message
        # here, we use top_k compressor on GPU
        # one can define more in compressors.py
        self.send_buffer = self.x - self.x_hat
        # values, indices = get_top_k(self.send_buffer.cuda(), self.ratio)
        values, indices = get_top_k(self.send_buffer, self.ratio)
        toc = time.time()

        values, indices = values.cpu(), indices.cpu()
        self.compressed = {"values":values, "indices":indices}

        return toc - tic



    def averaging(self, active_flags):
        self.comm.barrier()
        tic = time.time()

        # 根据激活的拓扑结构进行分散平均
        degree = 0
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    # Receive neighbor's message q_j
                    self.recv_tmp = self.comm.sendrecv(self.compressed, source=neighbor_rank, dest = neighbor_rank)
                    # Update aggregated model s += sum w_ij q_j
                    self.s[self.recv_tmp["indices"]] += self.neighbor_weight * self.recv_tmp["values"]

        # Compute self weight
        selfweight = 1 - degree * self.neighbor_weight
        # Update aggregated model s += w_ii q_i
        self.s[self.compressed["indices"]] += selfweight * self.compressed["values"]
        # Update x_hat = x_hat + q_i
        self.x_hat[self.compressed["indices"]] += self.compressed["values"]
        # Update local model parameters: x = x + consensus_lr*(s-x_hat)
        self.x.add_(self.s, alpha=self.consensus_lr).sub_(self.x_hat, alpha=self.consensus_lr)
        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.x,# .cuda(),
                            self.tensor_list),
                            self.tensor_list):
                t.copy_(f)

    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocess
        # 需要额外的编码时间
        encode_time = self.prepare_comm_buffer()

        # decentralized averaging
        # record the communication time
        comm_time = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return encode_time + comm_time