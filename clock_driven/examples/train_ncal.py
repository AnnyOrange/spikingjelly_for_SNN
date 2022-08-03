import argparse

import VGG
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101
import torch
import spikingjelly
import torch.nn as nn
from Trainer import Trainer
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
from torchvision.transforms import Compose
import transforms
import numpy as np
import random
seed = 2003
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("seed setup:{}".format(seed))


class BackgroundLoader(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


parser = argparse.ArgumentParser(description="train")
parser.add_argument("-T", default=14, type=int, help="simulating time-step")
parser.add_argument("-device", default="cuda:0", help="device")
parser.add_argument("-b", default=32
                    , type=int, help="batch size")
parser.add_argument("-lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("-datasetdir", default="./", type=str, help="dataset directory")
parser.add_argument("-resume", type=str, help="resume from file")
parser.add_argument("-opt", default="Adam", type=str, help="Adam or SGD optimizer")
parser.add_argument("-amp", default=True, type=bool, help="enable amp or not")
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
parser.add_argument('-step_size', default=20, type=float, help='step_size for StepLR')
parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')
parser.add_argument('-j', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()
log_file = "./Train_VGG_NCAL_LATAttention_T14.log"
def log(x):
    file = open(log_file, 'a')
    file.write(x+'\n')
    print(x)
    file.close()
def train_VGG_NCAL():
    if not args.resume:
        train_set, validate_set = \
            spikingjelly.datasets.split_to_train_test_set(0.9, NCaltech101(args.datasetdir, data_type='frame',
                                                                          frames_number=args.T,
                                                                          split_by='number',
                                                                          transform=transforms.RandomCompose(
                                                                              select_num=1,
                                                                              const_transforms=[transforms.Flip()],
                                                                              random_transforms=(
                                                                                  transforms.Rotation(15),
                                                                                  transforms.XShear(),
                                                                                  transforms.Cutout(),
                                                                                  transforms.Rolling()), )),
                                                          101)
        train_loader = BackgroundLoader(
            dataset=train_set,
            batch_size=args.b,
            shuffle=True,
            num_workers=args.j,
            drop_last=True,
            pin_memory=True
        )
        validate_loader = BackgroundLoader(
            dataset=validate_set,
            batch_size=args.b,
            shuffle=True,
            num_workers=args.j,
            drop_last=True,
            pin_memory=True
        )
        net = VGG.VGG_NCAL(T=args.T, kernel_sizeTS=3, kernel_sizeSC=1, device=args.device).to(args.device)

        log("Train_VGG_NCAL_LATAttention_T14 on cuda:0")
        trainer = Trainer(name="Train_VGG_NCAL_LATAttention", net=net, train_data=train_loader,
                          validate_data=validate_loader, test_data=None, criterion="MSELoss",
                          device=args.device, optimizer="Adam", lr=args.lr, opt_args=(),
                          tensorboard_path="./tensorboard_NCAL_T14", enable_amp=args.amp, lr_scheduler_name='StepLR')
        trainer.trian(500, 101, log, "./saves_NCAL_T14", 20, lr_scheuler_arg={'step_size': args.step_size})


if __name__ == '__main__':
    train_VGG_NCAL()