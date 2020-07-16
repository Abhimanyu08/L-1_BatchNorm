import torch
from torch import nn
from torch import tensor
import math

def _norm(x,dim,p):
    '''This function calculates the norm along all the 
    dimensions except the dimension given in dim argument'''
    shape = x.shape
    out_size = (1,)*dim + (x.size(dim),)+ (1,)*(x.dim()-1-dim)
    a = x.transpose(0,dim)
    b = a.contiguous().view(shape[dim], -1)
    xn = torch.norm(b, dim=1,p=p)
    xn = xn.view(*out_size)
    return xn/(shape[0]*shape[2]*shape[3])

torch.Tensor.l1norm = _norm

class BatchNorm1d(nn.Module):
    def __init__(self,nc,mom = 0.1,eps = 1e-05):
        super().__init__()
        self.mom = mom
        self.eps = eps
        self.mults = nn.Parameter(torch.ones(1,nc,1,1))
        self.adds = nn.Parameter(torch.zeros(1,nc,1,1))
        self.register_buffer('means', torch.zeros(1,nc,1,1))
        self.register_buffer('vars', torch.ones(1,nc,1,1))
        
    def update_stats(self,x):
        m = x.mean(dim = (0,2,3), keepdim = True)
        v = torch.abs(x-m).l1norm(dim =1, p=1)*(tensor(math.pi/2).sqrt())
        self.means.lerp_(m,self.mom)
        self.vars.lerp_(v,self.mom)
        return m,v
    
    def forward(self,x):
        if self.training: mean,var = self.update_stats(x)
        else:
            mean,var = self.means, self.vars
        x = (x - mean)/(var + self.eps)
        return x*self.mults + self.adds
