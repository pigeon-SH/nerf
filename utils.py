import torch
import numpy as np

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def positional_encoding(x, L=10):
    """ x shape: (batch_size, coarse or fine num, 3)
    """
    pi = torch.full((x.shape[0], x.shape[1], L), np.pi, device=device)
    exp = torch.arange(0, L, device=device).repeat((x.shape[0], x.shape[1], 1))
    twos = torch.full((x.shape[0], x.shape[1], L), 2, device=device)
    twos = torch.pow(twos, exp)
    encoded = torch.multiply(twos, pi)

    x = torch.repeat_interleave(x, repeats=L, dim=-1)
    encoded = torch.repeat_interleave(encoded, repeats=3, dim=-1)

    pe = torch.repeat_interleave(x, repeats=2, dim=-1)
    pe[:, :, ::2] = torch.sin(encoded * x)
    pe[:, :, 1::2] = torch.cos(encoded * x)
    return pe

def dist(x1, x2):
    return torch.linalg.norm(x1 - x2, 2)

def volume_rendering(x, sigma, rgb):
    """ x shape: (coarse_num, 3)
        sigma shape: (coarse_num, 1)
        rgb shape: (coarse_num, 3)
    """
    delta = torch.sqrt(torch.sum(torch.square(x[1:, :] - x[:-1, :]), dim=1, keepdim=True))    # shape: (coarse_num - 1, 1)
    delta = torch.cat([delta, torch.zeros_like(delta[0:1, :])], dim=0)    # shape: (coarse_num, 1) there is no distance for last element
    cs = torch.cumsum(delta * sigma, dim=0)
    T = torch.exp(-cs)
    weights = T * (1 - torch.exp(-delta * sigma))
    C = torch.sum(weights * rgb)
    return C, weights


