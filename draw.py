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


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
