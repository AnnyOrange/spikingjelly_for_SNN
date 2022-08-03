import torch
import torch.nn as nn
from Experiment import Argument, CIFAR10DVSVGGExperiment, NCALTECH101VGGExperiment

arg_cifar_tet = Argument()
arg_cifar_tet.criterion = 'MSELoss'
arg_cifar_tet.T = 14
arg_cifar_tet.device = "cuda:0"
arg_cifar_tet.lr_scheduler = "StepLR"
arg_cifar_tet.epoch = 250
arg_cifar_tet.rank = 32
cifar_experiment = CIFAR10DVSVGGExperiment(name="CIFAR10VGG_T14_rank32", args=arg_cifar_tet)
cifar_experiment.train()

print("-----------------------------CIFAR10VGG_T14_rank32---------------------------------------------")