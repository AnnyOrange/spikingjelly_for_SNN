import torch
import torch.nn as nn
from spikingjelly.clock_driven.surrogate import SurrogateFunctionBase, heaviside


class triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.require_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = (1 / ctx.alpha) * (1 / ctx.alpha) * ((ctx.alpha - ctx.saved_tensors[0].abs()).clamp(dim=0))
            grad_x = grad_x * grad_output
        return grad_x, None
        # (input, out, others) = ctx.saved_tensors
        # alpha = others[0].item()
        # grad_input = grad_output.clone()
        # tmp = (1 / alpha) * (1 / alpha) * ((alpha - input.abs()).clamp(min=0))
        # grad_input = grad_input * tmp
        # return grad_input, None


class Triangle(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return triangle.apply(x, alpha)

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplemented("No primitive function for Triangle surrogate function")