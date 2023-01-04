import sys
import torch
import d2lzh_pytorch as d2l
from torch import nn
import torchvision.transforms as transforms
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w] * K).sum()
    return Y


def loss_plot(step, loss, x_label='Step', y_label='Loss', x_limit=1):
    d2l.set_figsize(figsize=(5, 2.5))
    plt.xticks(step[::x_limit])
    d2l.plt.plot(step, loss)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
