import torch
import torch.nn as nn
import numpy as np
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()

        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads)
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_Dropkey = True
        self.mask_ratio = 0.15

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='floor')).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_Dropkey == True:
            m_r=torch.ones_like(attn) * self.mask_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12


        attn = attn.softmax(dim=-1)

        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class JAttention(nn.Module):
    def __init__(self, in_channels, out_channels=2, dropout=0.):
        super().__init__()
        self.emb = nn.Linear(in_channels, 256)
        self.pa=nn.Parameter(torch.zeros([1, 17, 256]))
        self.L = nn.Linear(256, 256)
        self.ATT = Attention(256)
        self.MLP = Mlp(256, 512, 256 , dropout = dropout)
        self.fc = nn.Linear(256, out_channels)


    def forward(self, x):
        x = self.emb(x)
        x = x + self.pa
        
        x = self.L(x)
        x = self.ATT(x)
        x = self.MLP(x)
        x = self.fc(x)
        return x