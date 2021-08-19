import torch
from torch.nn import functional as F, Parameter, Module
from typing import Dict

def hessian(x:torch.Tensor, kernel:torch.Tensor):
    '''
    Calculate the hessian matrix per channel given
    a kernel to calculate the horizontal gradient
    '''
    B, C, H, W = x.shape
    # B * C, 2, H, W
    y = F.conv2d(
        x.reshape(B * C, 1, H, W),
        kernel
    )
    H, W = y.shape[2:4]
    y = y.view(B * C * 2, 1, H, W)
    y = F.conv2d(
        y,
        kernel
    )
    H, W = y.shape[2:4]
    return y.view(B, C, 2, 2, H, W)

class Hessian(Module):
    '''
    Module to calculate the hessian matrix per channel given
    a kernel to calculate the horizontal gradient or
    construct a sobel kernel with given kernel_size
    '''
    def __init__(self, kernel:torch.Tensor=None, kernel_size:int=2, **conv_kwargs):
        '''
        `kernel` is used to calculate the horizontal gradient or
        construct a sobel kernel with given `kernel_size`.
        `conv_kwargs` are forwarded to `conv2d`
        '''
        super().__init__()
        self.conv_kwargs = conv_kwargs
        if kernel is None:
            from .kernel import sobel
            kernel = torch.from_numpy(sobel(kernel_size)).to(torch.float32)
        self.weight = Parameter(torch.stack((kernel, kernel.t()))[:, None], requires_grad=False)

    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        # B * C, 2, H, W
        y = F.conv2d(
            x.reshape(B * C, 1, H, W),
            self.weight,
            **self.conv_kwargs
        )
        H, W = y.shape[2:4]
        y = F.conv2d(
            y.view(B * C * 2, 1, H, W),
            self.weight,
            **self.conv_kwargs
        )
        H, W = y.shape[2:4]
        return y.view(B, C, 2, 2, H, W)
