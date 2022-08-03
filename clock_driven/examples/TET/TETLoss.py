import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


def TET_loss(outputs, labels, criterion, means, lamb):
    # output: TxNx...
    T = outputs.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[t], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total


class TETLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, criterion, means=1.0, lamb=0.001, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super(TETLoss, self).__init__(size_average, reduce, reduction)
        self.criterion = criterion
        self.means = means
        self.lamb = lamb

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return TET_loss(input, target, self.criterion, self.means, self.lamb)