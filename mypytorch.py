import sys
import torch
import d2lzh_pytorch as d2l
from torch import nn
import torchvision
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
from IPython import display
from matplotlib import pyplot as plt
import zipfile
import numpy as np
import random
import math


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def loss_plot(step, loss, x_label='Step', y_label='Loss', x_limit=1):
    d2l.set_figsize(figsize=(5, 2.5))
    plt.xticks(step[::x_limit])
    d2l.plt.plot(step, loss)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)


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


def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


def corr2d_multi_in_out_1x1(X, K):
    c_o, c_i = K.shape[0: 2]
    h, w = X.shape[1: 3]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    res = torch.mm(K, X).view(c_o, h, w)
    return res


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = (X[i: i + p_h, j: j + p_w]).max()
            elif mode == 'avg':
                Y[i, j] = (X[i: i + p_h, j: j + p_w]).mean()
    return Y


# ?????????????????????d2lzh_pytorch?????????????????????????????????????????????????????????
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # ???????????????device?????????net???device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # ????????????, ????????????dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # ??????????????????
            else:  # ??????????????????, 3.13?????????????????????, ?????????GPU
                if 'is_training' in net.__code__.co_varnames:  # ?????????is_training????????????
                    # ???is_training?????????False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epoch):
    net = net.to(device)  # ???????????????device?????????
    print('training on', device)
    loss = nn.CrossEntropyLoss()
    l_lst = []
    for epoch in range(num_epoch):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for i, (X, y) in enumerate(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        l_lst.append(train_l_sum / batch_count)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return l_lst


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    print('?????????????????????:', len(mnist_train), '?????????????????????:', len(mnist_test))

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter


class GlobalAvgPool2d(nn.Module):
    # ????????????????????????????????????????????????????????????????????????????????????
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        output = F.relu(X + Y)
        return output


class FlattenLayer(nn.Module):  # ???????????????
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def load_data_jay_lyrics(num_char):
    """
    :return: corpus_chars, idx_to_char, vocab_size, corpus_indices
    """
    with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:num_char]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    """
    :param corpus_indices: ??????????????????
    :param batch_size: ????????????
    :param num_steps: ????????????
    :param device: ??????
    :return: ????????????
    """
    # ???1????????????????????????x????????????????????????y???1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # ?????????pos???????????????num_steps?????????
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # ????????????batch_size???????????????
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """
    :param corpus_indices: ??????????????????
    :param batch_size: ????????????
    :param num_steps: ????????????
    :param device:None ??????
    :return: ????????????
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_len * batch_size].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        j = i * num_steps
        X = indices[:, j: j + num_steps]
        Y = indices[:, j + 1: j + num_steps + 1]
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros((x.shape[0], n_class), dtype=dtype)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(x, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(x[:, i], n_class) for i in range(x.shape[1])]


def _one(shape, device):
    ts = torch.tensor(np.random.normal(0, 0.01, size=shape), dtype=torch.float32, device=device)
    return nn.Parameter(ts, requires_grad=True)


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # inputs???outputs??????num_steps????????????(batch_size, vocab_size)?????????
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    """
    ????????????prefix?????????num_chars???????????????\n
    :param prefix:??????
    :param num_chars:???????????????
    :param rnn:
    :param params:
    :param init_rnn_state:
    :param num_hiddens:
    :param vocab_size:
    :param device:
    :param idx_to_char:
    :param char_to_idx:
    :return:num_chars???????????????
    """
    state = init_rnn_state(1, num_hiddens, device)
    outputs = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(torch.tensor([[outputs[-1]]], device=device), vocab_size)
        Y, (state,) = rnn(X, state, params)
        if t < len(prefix) - 1:
            outputs.append(char_to_idx[prefix[t + 1]])
        else:
            outputs.append(int(Y[0].argmax(dim=1).item()))
    return ''.join(idx_to_char[i] for i in outputs)


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:  # ??????????????????
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params(vocab_size, num_hiddens, vocab_size)  # ???????????????
    loss = nn.CrossEntropyLoss()  # ?????????????????????

    for epoch in range(num_epochs):
        if not is_random_iter:  # ???????????????????????????epoch??????????????????????????????
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # ????????????????????????????????????????????????????????????????????????
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # ??????????????????detach????????????????????????????????????, ????????????
                # ???????????????????????????????????????????????????????????????????????????(??????????????????????????????)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs???num_steps????????????(batch_size, vocab_size)?????????
            (outputs, state) = rnn(inputs, state, params)
            # ?????????????????????(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y????????????(batch_size, num_steps)??????????????????????????????
            # batch * num_steps ?????????????????????????????????????????????
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)  # transpose?????????0???1?????????, contiguous????????????????????????
            # ?????????????????????????????????????????????
            l = loss(outputs, y.long())

            # ?????????0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # ????????????
            d2l.sgd(params, lr, 1)  # ?????????????????????????????????????????????????????????
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                        num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs:(batch_size, seq_len)
        X = to_onehot(inputs, self.vocab_size)  # X:(seq_len, batch_size, vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)  # Y:(seq_len, batch_size, num_hiddens)
        outputs = self.dense(Y.view(-1, Y.shape[-1]))  # outputs:(seq_len*batch_size, vocab_size)
        return outputs, state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output?????????prefix????????????
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


# ?????????????????????d2lzh_pytorch????????????????????????
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                                  prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    perplexity_lst = []
    pred_list = []
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # ????????????
        for X, Y in data_iter:
            if state is not None:
                # ??????detach????????????????????????????????????, ????????????
                # ???????????????????????????????????????????????????????????????????????????(??????????????????????????????)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state)  # output: ?????????(num_steps * batch_size, vocab_size)

            # Y????????????(batch_size, num_steps)??????????????????????????????
            # batch * num_steps ?????????????????????????????????????????????
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())  # ?????????????????????

            optimizer.zero_grad()
            l.backward()
            # ????????????
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            perplexity_lst.append(perplexity)
            pred_list.append(epoch + 1)
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))
        plot(pred_list, perplexity_lst, xlabel='epoch', ylabel='perplexity')


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1???s2????????????????????????????????????????????????
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


# ?????????????????????d2lzh_pytorch????????????????????????
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
