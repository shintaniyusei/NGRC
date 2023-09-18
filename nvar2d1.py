import itertools as it
import profile
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class NVAR(nn.Module):
    def __init__(self, k, skip, transients, alpha):
        super(NVAR, self).__init__()
        self.k = k
        self.skip = skip
        self.transients = transients
        self.alpha = alpha

    def forward(self, X, p=3, bias=True, window=None):
        X = X.unsqueeze(3)
        batch_size, row, n_steps, n_dim = X.shape

        lin_dim = n_dim * self.k
        lin_idx = torch.arange(lin_dim).to(device)
        monom_idx = torch.tensor(list(it.combinations_with_replacement(lin_idx, p))).to(device)

        nlin_dim = monom_idx.shape[0]
        win_dim = (self.k - 1) * self.skip + 1
        if window is None:
            window = torch.zeros((batch_size, row, win_dim, n_dim)).to(device)

        lin_features = torch.zeros((batch_size, row, n_steps, lin_dim)).to(device)
        nlin_features = torch.zeros((batch_size, row, n_steps, nlin_dim)).to(device)
        for i in range(n_steps):
            window = torch.roll(window, -1, 2)
            window[:,:, -1, :] = X[:,:,i,:]

            lin_feat_ = window[:,:, ::self.skip, :]
            lin_feat = lin_feat_.flatten(2)

            lin_features[:,:,i,:] = lin_feat

        lin_feat_monom = lin_features[:,:,:,monom_idx]
        nlin_features = torch.prod(lin_feat_monom, axis=4)
        n_steps = lin_features.shape[2] - self.transients
        tot_features_ = torch.cat((lin_features, nlin_features), 3).to(device)
        tot_features = tot_features_[:,:,self.transients:,:]
        c = torch.ones((batch_size, row, n_steps, 1)).to(device)
        tot_features = torch.cat((c, tot_features), 3).squeeze(3)

        return tot_features.to(device)
    

    
class NVAR2D(nn.Module):
    def __init__(self, k, skip, transients, alpha):
        super(NVAR2D, self).__init__()
        self.k = k
        self.skip = skip
        self.transients = transients
        self.alpha = alpha
        self.NVAR = NVAR(self.k, self.skip, 9, alpha=1e-6)

        self.fc = nn.LazyLinear(128)
        self.fc_v = nn.LazyLinear(49)
        self.fc_h = nn.LazyLinear(128)
        self.layer_norm1 = nn.LayerNorm(128, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(128, eps=1e-6)
        
    def forward(self, x):
        x = self.layer_norm1(x)
        v = x.permute(0, 2, 1)
        v = self.NVAR(v)
        v = v.flatten(2)
        v = self.fc_v(v)
        v = v.permute(0, 2, 1)

        v = v + x
        v = self.layer_norm2(v)

        h = self.NVAR(x)
        h = h.flatten(2)
        h = self.fc_h(h) + x
        
        x = torch.cat([v, h], dim=-1)
        x = self.fc(x)
        return x.to(device)
