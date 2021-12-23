import torch
import numpy as np

from torch.nn import functional as F


def linear(inputs, weight, bias):
    if bias is None:
        return F.linear(inputs, weight.cuda(), None)
    else:
        return F.linear(inputs, weight.cuda(), bias.cuda())

def conv2d(inputs, weight, bias, stride, padding):
    if bias is None:
        return F.conv2d(inputs, weight.cuda(), None, stride, padding)
    else:
        return F.conv2d(inputs, weight.cuda(), bias.cuda(), stride, padding)

def maxpool(inputs, kernel_size, stride):
    return F.max_pool2d(inputs, kernel_size, stride)

def batchnorm(inputs, weight, bias, device, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    running_mean = torch.zeros(np.prod(np.array(inputs.data.size()[1]))).cuda()
    running_var = torch.ones(np.prod(np.array(inputs.data.size()[1]))).cuda()
    return F.batch_norm(inputs, running_mean, running_var, weight, bias, training, momentum, eps)
