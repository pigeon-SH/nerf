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

def volume_rendering(t, sigma, rgb):
    """ t shape: (batch_size, coarse_num, 1)
        sigma shape: (batch_size, coarse_num, 1)
        rgb shape: (batch_size, coarse_num, 3)
    """
    delta = t[:, 1:, :] - t[:, :-1, :]  # shape: (batch_size, coarse_num-1, 1)
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[:, 0:1, :])], dim=1)    # shape: (batch_size, coarse_num, 1), inf for last distance
    cs = torch.cumsum(delta * sigma, dim=1) # shape: (batch_size, coarse_num, 1)
    T = torch.exp(-cs)
    weights = T * (1. - torch.exp(-delta * sigma))  # shape: (batch_size, coarse_num, 1)
    C = torch.sum(weights.clone() * rgb.clone(), dim=1) # shape: (batch_size, 1)
    return C, weights


