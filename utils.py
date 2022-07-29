import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encoding_function(p, L=10):
    """ p: scalar value
        return (sin(2^0 pi p), cos(2^0 pi p), ...)
    """
    encoded = np.ones((2 * L))
    for i in range(2 * L):
        if i % 2 == 0:
            encoded[i] = np.sin(2 ** (i // 2) * np.pi * p)
        else:
            encoded[i] = np.cos(2 ** (i // 2) * np.pi * p)
    
    return torch.tensor(encoded).to(device=device, dtype=torch.float32)

def positional_encoding(x, L=10):
    """ x: (x, y ,z) or (theta, phi, 1) tuple
        return result of positional encoding
    """
    pe = []
    for p in x:
        pe.append(encoding_function(p.item(), L))
    
    return torch.cat(pe)

def dist(x1, x2):
    return torch.linalg.norm(x1 - x2, 2)

def volume_rendering(x, sigma, rgb):
    """ x shape: (coarse_num, 3)
        sigma shape: (coarse_num, 1)
        rgb shape: (coarse_num, 3)
    """
    N = len(x)
    C = 0
    weights = []
    for i in range(N):
        s = torch.tensor([0.]).to(device=device, dtype=torch.float32)
        for j in range(i - 1):
            delta = torch.dist(x[j + 1], x[j])
            s += delta * sigma[j]
        T = torch.exp(-s)
        if i < N - 1:
            weight = T * (1 - torch.exp(-sigma[i] * (torch.dist(x[i + 1], x[i]))))
        else:
            weight = torch.tensor([0.]).to(device=device, dtype=torch.float32)
        weights.append(weight)
        C += weight * rgb[i]
    weights = torch.cat(weights, dim=0)
    return C, weights


