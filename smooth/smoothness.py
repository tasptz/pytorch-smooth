import torch
from .hessian import Hessian

class Smoothness(Hessian):
    '''
    Module to calculate a smoothness value per channel
    (negative mean squared sum of hessian matrix per pixel).
    '''

    def forward(self, x:torch.Tensor):
        h = super().forward(x)
        s = torch.pow(h, 2)
        # spatial dimensions
        s = s.mean((-1, -2))
        # hessian dimensions
        s = s.sum((-1, -2))
        return -s
