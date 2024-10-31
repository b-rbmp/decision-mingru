import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Union
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


# just for compatibility with other files, we do not use this Convolution module
class Convolution(nn.Module):
    def __init__(self, config, hidden_size, block_index):
        super().__init__()
        self.window_size = config.window_size
        self.remove_act_embs = config.remove_act_embs

        self.rtg_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.obs_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        if not self.remove_act_embs:
            self.act_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)

    def forward(self, x):
        #window_size = self.window_size

        # pad the input tensor with zeros along the sequence dimension
        padded_tensor = torch.nn.functional.pad(x, (0, 0, self.window_size - 1, 0)).transpose(1, 2)

        if not self.remove_act_embs:
            rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
            obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
            act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

            conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3), act_conv_tensor.unsqueeze(3)), dim=3)
            conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)

        else:
            rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::2]
            obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::2]

            conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3)), dim=3)
            conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)

        conv_tensor = conv_tensor.transpose(1, 2)  #.to('cuda')
        return conv_tensor







class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
#********** ********* ********** **********

