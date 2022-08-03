import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class self_attention_T(nn.Module):
    def __init__(self, T: int , H:int , W: int , C: int , kernel_size: int=3,device='cpu'):
        super(self_attention_T,self).__init__()
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.N = None
        self.D = 3 #随便设的
        # self.conv1d = nn.Conv1d(in_channels=self.C*self.W*self.H,out_channels = self.D,
        #                         kernel_size=2,stride=1,padding=0,
        #                         bias = False)
        # self.project1 = nn.Linear(self.C*self.W*self.H, self.D)
        # self.project2 = nn.Linear(self.D, self.C*self.W*self.H)
        '''如果是映射，而且要映射回去，那是不是矩阵更好呢，我生成一个C矩阵[CHW,D],
        这样映射过去和映射回去遵守的法则有一定的关系'''
        self.W_matrix = torch.nn.Parameter(torch.ones(self.C*self.W*self.H,self.D))
        self.W_matrix1 = self.W_matrix.T
        self.query = nn.Linear(self.D,self.D)
        self.key = nn.Linear(self.D, self.D)
        self.value = nn.Linear(self.D, self.D)

    def forward(self, x):
        #[T,N,C,H,W]
        x = x.permute(1, 0, 2, 3, 4) # [N,T,C,H,W]
        [self.N, self.T, self.C, self.H, self.W] = x.size()
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]*x.shape[4]) #[N,T,C*H*W]
        # x = x.permute(0,2,1)
        # x = self.conv1d(x)
        # x = x.permute(0,2,1)#[N,T,D] 下一步就是写成QKV做自注意力
        x = torch.matmul(x,self.W_matrix)#[N,T,D]
        Q = self.query(x)  #[N,T,D]
        K = self.key(x)
        V = self.value(x)
        d_k = Q.shape[1]*Q.shape[2] # d_k = T*D
        scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k) #[N,T,T]
        scores = torch.nn.functional.softmax(scores,dim = -1)
        y = torch.matmul(scores,V) #[N,T,D]
        y = torch.matmul(y,self.W_matrix1) #[N,T,CHW]
        y = y.reshape(self.N, self.T, self.C, self.H, self.W)
        y = y.permute(1, 0, 2, 3, 4)#[T,N,C,H,W]
        return y

# class PEG(nn.Module):
#     def __init__(self,dim: int,kernel_size: int):
#         super().__init__()
#         self.conv2d_zero(dim, dim, kernel_size, 1,kernel_size//2,groups = dim)
#     def forward(self,x:torch.Tensor):
#         N,T,D = x.shape
#         x = x.transpose(1,2).view(B,C,H,W)
#         x = self.conv2d_zero(x)+x
#         x = x.flatten(2).transpose(1, 2)
#         x = torch.cat(())
#         return x


def main():
    print("start")
    x = torch.randn(2,3,5,3,4)#[T,N,C,H,W]  #随便设的
    print("x:\n",x.shape)
    model = self_attention_T(T = 2,H = 3,W = 4,C = 5)
    y = model(x)
    print(y.shape)



if __name__ == '__main__':
    main()