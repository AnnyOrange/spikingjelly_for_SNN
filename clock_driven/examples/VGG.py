import torch
import torch.nn as nn

import local_add_T_attention
from spikingjelly.clock_driven import surrogate, layer, neuron, functional
import torchstat

import TET.Triangle


class VGG_CIFAR(nn.Module):  # test origin modlue
    def __init__(self, T: int = 14,kernel_size = 3,rank = 64, device='cpu'):
        super().__init__()
        conv = []
        conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
        conv.append(
            local_add_T_attention.L_T_attention(T1=T, C=2,H=48, W=48))
        conv.extend(VGG_CIFAR.conv3x3(2, 64))
        conv.extend(VGG_CIFAR.conv3x3(64, 128))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention(T=T, C=128, H=24, W=24))

        conv.extend(VGG_CIFAR.conv3x3(128, 256))
        conv.extend(VGG_CIFAR.conv3x3(256, 256))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention( T=T, C=256, H=12, W=12))

        conv.extend(VGG_CIFAR.conv3x3(256, 512))
        conv.extend(VGG_CIFAR.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention(T=T, C=512,H=6, W=6))

        conv.extend(VGG_CIFAR.conv3x3(512, 512))
        conv.extend(VGG_CIFAR.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention( T=T, C=512,H=3, W=3))

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * 3 * 3, 10)),
            neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 10]
        return out_spikes

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
        ]
class VGG_NCAL(nn.Module):  # test origin modlue
    def __init__(self, T: int = 14, kernel_size=3, rank=64, device='cpu'):
        super().__init__()
        conv = []
        conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
        conv.append(
            local_add_T_attention.L_T_attention(T1=T, C=2,H=48, W=48))
        conv.extend(VGG_NCAL.conv3x3(2, 64))
        conv.extend(VGG_NCAL.conv3x3(64, 128))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention(T=T, C=128, H=24, W=24))

        conv.extend(VGG_NCAL.conv3x3(128, 256))
        conv.extend(VGG_NCAL.conv3x3(256, 256))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention( T=T, C=256, H=12, W=12))

        conv.extend(VGG_NCAL.conv3x3(256, 512))
        conv.extend(VGG_NCAL.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention(T=T, C=512,H=6, W=6))

        conv.extend(VGG_NCAL.conv3x3(512, 512))
        conv.extend(VGG_NCAL.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            local_add_T_attention.L_T_attention( T=T, C=512,H=3, W=3))

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * 3 * 3, 101)),
            neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 101]
        return out_spikes

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
        ]

