import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist
import functools


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
        将梯度从不同部分的神经网络参数展平并连接，以便进行梯度更新操作（如优化算法中的梯度下降）或者用于某些分布式计算操作
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
        这个函数的作用是从一个1D缓冲区中还原多个张量，并确保它们的大小正确匹配。这在分布式深度学习中，特别是在梯度聚合等操作中非常有用。
        通常，它与 flatten_tensors 函数一起使用，前者用于将多个张量合并成一个1D缓冲区，后者用于将1D缓冲区还原为多个张量。
        这有助于在分布式环境中传输和同步模型参数或梯度。
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def communicate(tensors, communication_op):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    在分布式环境中，不同计算节点上的模型参数或梯度需要进行同步和协调以保持模型的一致性。
    这个函数的设计允许您使用不同的通信操作（例如 Allreduce）来执行不同的分布式通信。
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)

