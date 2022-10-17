# 具体的思路是见传入的[N,T,C,H,W]换成[NT,C,H,W]换成[NT,D,H,W]用swin
# 然后[N,T,HWC]之后关于T做attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from Swin import T


class L_T_attention(nn.Module):
    def __init__(self,T1: int, C:int, H: int, W:int):
        super().__init__()
        self.T1 = T1
        self.N = None
        self.H = H
        self.W = W
        self.D = 32
        self.C = C

        self.project_in = nn.Linear(self.C,self.D)
        self.T_self = self_attention_T(P=8*self.H*self.W,C = self.C,H = self.H,W = self.W)
        self.T_self1 = self_attention_T(P=512, C=self.C, H=self.H, W=self.W)
        self.T_self2 = self_attention_T(P=2048, C=self.C, H=self.H, W=self.W)
        self.T_self3 = self_attention_T(P=8192, C=self.C, H=self.H, W=self.W)

        self.t1 = T(img_size=48, patch_size=1, in_chans=32, embed_dim=32, depths=[2, 4, 4], num_heads=[8, 8, 8],
                    window_size=4)

        self.t3 = T(img_size=16, patch_size=1, in_chans=32, embed_dim=32, depths=[2, 4, 4], num_heads=[8, 8, 8],
                    window_size=4)
        self.t4 = T(img_size=32, patch_size=1, in_chans=32, embed_dim=32, depths=[2, 4, 4], num_heads=[8, 8, 8],
                    window_size=4)
        self.t2 = T(img_size=8, patch_size=1, in_chans=32, embe_dim = 32, depths=[2, 4, 4], num_heads=[8, 8, 8],
                    window_size=4)
    def forward(self,x):

        #x [T,N,C,H,W]
        x = x.permute(1, 0, 2, 3, 4)  # [N,T,C,H,W]
        [self.N, self.T1, self.C, self.H, self.W] = x.size()
        # print(self.N)
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # [NT,C,H,W]
        x = x.permute(0,2,3,1) #[NT,H,W,C]
        x = (self.project_in(x)).permute(0,3,1,2) #[NT,H,W,D]不应该是[B,D,H,W]
        if (self.H == 48):
            x = self.t1(x) #[NT,128,H/4,W/4]
            x = x.reshape(self.N,self.T1, x.shape[1]*x.shape[2]*x.shape[3])
            x = self.T_self(x)
        if (self.H == 6):
            x = F.pad(x, pad=(1, 1, 1, 1), mode="constant", value=0)  # 将[B,D,H,W]转化为[B,D,8,8]
            x = self.t2(x)#[B,128,2,2]
            x = x.reshape(self.N, self.T1, x.shape[1] * x.shape[2] * x.shape[3])
            x = self.T_self1(x)
        if (self.H == 12):
            x = F.pad(x, pad=(2, 2, 2, 2), mode="constant", value=0)  # 将[B,D,H,W]转化为[B,D,16,16]
            x = self.t3(x)#【B,128,4,4】
            x = x.reshape(self.N, self.T1, x.shape[1] * x.shape[2] * x.shape[3])
            x = self.T_self2(x)
        if (self.H == 24):
            x = F.pad(x, pad=(4, 4, 4, 4), mode="constant", value=0)#将[B,D,H,W]转化为[B,D,32,32]
            x = self.t4(x)#[B,128,8,8]
            x = x.reshape(self.N, self.T1, x.shape[1] * x.shape[2] * x.shape[3])
            x = self.T_self3(x)
        # x = self.T_self(x)
        return x


class self_attention_T(nn.Module):
    def __init__(self, P :int,C: int,H: int,W: int):
        super().__init__()
        self.D = 32 #随便设的
        self.P = P
        self.C = C
        self.H = H
        self.W = W
        # self.conv1d = nn.Conv1d(in_channels=self.C*self.W*self.H,out_channels = self.D,
        #                         kernel_size=2,stride=1,padding=0,
        #                         bias = False)
        # self.project1 = nn.Linear(self.C*self.W*self.H, self.D)
        # self.project2 = nn.Linear(self.D, self.C*self.W*self.H)
        # '''如果是映射，而且要映射回去，那是不是矩阵更好呢，我生成一个C矩阵[CHW,D],
        # 这样映射过去和映射回去遵守的法则有一定的关系'''
        self.project_in = nn.Linear(self.P,self.D)
        self.project_out = nn.Linear(self.D,self.C*self.H*self.W)
        # self.W_matrix = torch.nn.Parameter(torch.ones(self.C*self.W*self.H,self.D))
        # self.W_matrix1 = self.W_matrix.T
        self.query = nn.Linear(self.D,self.D)
        self.key = nn.Linear(self.D, self.D)
        self.value = nn.Linear(self.D, self.D)

    def forward(self, x):
        [self.N, self.T, self.P] = x.size()
        x = self.project_in(x) #[N,T,D]
        # x = torch.matmul(x,self.W_matrix)#[N,T,D]
        Q = self.query(x)  #[N,T,D]
        K = self.key(x)
        V = self.value(x)
        d_k = Q.shape[1]*Q.shape[2] # d_k = T*D
        scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k) #[N,T,T]
        scores = torch.nn.functional.softmax(scores,dim = -1)
        y = torch.matmul(scores,V) #[N,T,D]
        # y = torch.matmul(y,self.W_matrix1) #[N,T,CHW]
        y = self.project_out(y)#[N,T,CHW]
        y = y.reshape(self.N, self.T, self.C, self.H, self.W)
        y = y.permute(1, 0, 2, 3, 4)#[T,N,C,H,W]
        return y

# net = L_T_attention(T1=10,C = 64,H = 128,W = 128)
#
# x = torch.randn(10,64,64,128,128)
# x = net(x)
# print(x.shape)

def main():
    device = torch.device('cuda:0')
    print("start")
    net = L_T_attention(T1=2,C = 2,H = 24,W = 24).to(device)
    x = torch.randn(2, 4, 2, 24, 24).to(device)
    print("x:\n",x.shape)

    y = net(x).to(device)
    print(y.shape)
    #[B,D,H,W]

if __name__ == '__main__':
    main()