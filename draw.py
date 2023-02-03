import d2lzh_pytorch as d2l
from matplotlib import pyplot as plt
import numpy as np


def plot(x, y, xlabel='x', ylabel='y', xlim=None, ylim=None, xticks=None, yticks=None, figsize=(5, 2.5),
         other_funcs=None):
    d2l.set_figsize(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xticks:
        plt.xticks(x[::xticks])
    if yticks:
        plt.yticks(y[::yticks])
    if other_funcs:
        for func in other_funcs:
            func[0](func[1])
    plt.plot(x, y)


def multiplot(*args, xlabel='x', ylabel='y', xlim=None, ylim=None, figsize=(5, 2.5)):  # [[x, y, label], [x, y, label]]
    plt.rcParams['figure.figsize'] = figsize
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    for line in args:
        plt.plot(line[0], line[1], label=line[2])
    plt.legend()



