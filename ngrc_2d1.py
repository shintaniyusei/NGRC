import itertools as it
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nvar2d1 import NVAR2D


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatchAndPosEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=128, drop_out=0.1):
        super(PatchAndPosEmbedding, self).__init__()

        num_patches = int((img_size/patch_size)**2)
        patch_size_dim = patch_size*patch_size*in_channels

        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=patch_size_dim, kernel_size=patch_size, stride=4)
        self.linear = nn.Linear(patch_size_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))   
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim)) 

        self.dropout = nn.Dropout(drop_out)

    def forward(self, img):
        x = self.patch_embedding(img) # [B,C,H,W] -> [B, patch_size_dim, N, N] # N = Num_patches = (H*W)/Patch_size,
        x = x.flatten(2)
        x = x.transpose(2, 1)  # [B,N*N, patch_size_dim]
        x = self.linear(x)     # [B,N*N, embed_dim]  # patch_size_dim -> embed_dim = 3072->1024 to reduce the computation when encode.
        out = self.dropout(x)

        return out


class MLP(nn.Module):  
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
def extract_patches(x, patch_size):
    # if kernel != 1:
    #     x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    all_patches = all_patches.flatten(4,5).flatten(1,2)
    all_patches = all_patches.flatten(2,3)
    # all_patches = all_patches.unsqueeze(3).permute(0, 1, 3, 2)
    return all_patches.to(device)

class NGRC(nn.Module):
    def __init__(self, k, skip, num_classes=10, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, dropout=0.1):
        super(NGRC, self).__init__()
        self.patch_size = patch_size
        self.patchembedding = PatchAndPosEmbedding(img_size, patch_size, in_channels, embed_dim, dropout)
        
        self.NVAR2D = NVAR2D(k, skip, 0, alpha=1e-6)
        
        self.layer_norm = nn.LayerNorm([49, embed_dim], eps=1e-6)
        self.linear = nn.LazyLinear(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim*2)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
    
    
    def forward(self, x):
        # x = extract_patches(x, self.patch_size)
        # x = self.linear(x)
        x = self.patchembedding(x)
        x = self.NVAR2D(x)
 
        x2 = self.layer_norm(x)
        x = self.mlp(x2)
        x = x.permute(0, 2, 1)
        x = torch.squeeze(F.dropout(self.gap(x), training=self.training), 2)
        
        x = F.relu(self.mlp_head(x))
        
        return F.log_softmax(x).to(device)
