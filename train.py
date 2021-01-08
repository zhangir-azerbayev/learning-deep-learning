# Contains a number of functions useful for training neural networks. 
import torch 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def softmax(X): 
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition 

def count_correct(y_hat, y): 
    """Compute the number of correct predictions"""
    if len(y_hat.shape)>1 and y_hat.shape[1]>1: 
        y_hat = y_hat.argmax(axis=1)
    comp = y_hat.type(y.dtype) == y
    return float(comp.sum())

class Accumulator: 
    """For accumulating sums over `n` distinct sums"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args): 
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

def evaluate_accuracy(net, data_iter): 
    """Evaluates the accuracy of a `torch.nn.Module` object over a 
    `torch.utils.data.dataloaderDataLoader` object. """
    net.eval()
    metric = Accumulator(2) # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(count_correct(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater): 
    """One epoch of training for a classifier. 
    `net` is a `torch.nn.Module`
    `train_iter` is a torch.utils.data.dataloader.DataLoader object
    `loss` is the cost function
    `updater` is `torch.optim.Optimizer`"""
    net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter: 
        y_hat = net(X)
        l = loss(y_hat, y)

        # Computes and updates parameters
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(float(l) * len(y), count_correct(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator: 
    """Plots training and test error at each epoch. 
    Updates in real-time if matplotlib is in notebook mode."""
    def __init__(self, legend=None, xlim=None): 
        if legend is None: 
            legend = []
        self.fig, self.axes = plt.subplots(1)
        self.legend = legend
        self.xlim = xlim
        self.X, self.Y = None, None

    def add(self, x, y): 
        if self.X is None: 
            self.X = x.numpy()
        else: 
            self.X = np.concatenate((self.X, x.numpy()))
        if self.Y is None: 
            self.Y = y.numpy()
        else: 
            self.Y = np.concatenate((self.Y, y.numpy()))
        self.axes.clear()
        self.axes.set(xlim = self.xlim, xlabel='epoch')
        self.axes.plot(self.X, self.Y)
        self.axes.legend(self.legend)
        self.fig.canvas.draw()


def train(net, train_iter, test_iter, loss, num_epochs, updater): 
    """Train a model"""
    graph = Animator(legend = ('train acc', 'test acc'), xlim = [1, num_epochs])
    for epoch in range(num_epochs): 
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        graph.add(torch.tensor([epoch + 1]), torch.stack(
            (torch.tensor([train_acc]), torch.tensor([test_acc])), dim=1))

    
