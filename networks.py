import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 256)   # 60: positional encoding 2*L=20 for each x, y, z
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256 + 60, 256) # skip connection
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256 + 1)  # sigma output
        self.add_fc1 = nn.Linear(256 + 24, 128) # 24: positional encoding 2*L=8 for each theta, phi, 1
        self.add_fc2 = nn.Linear(128, 3)    # rgb output
    
    def forward(self, x, d):
        """ x: result of positional encoding of location, shape=(batch_size, sample_num, 60)
            d: result of positional encoding of viewing direction, shape=(batch_size, sample_num, 24)
        """
        x = torch.flatten(x, 0, 1)
        d = torch.flatten(d, 0, 1)
        v = self.fc1(x)
        v = F.relu(v)
        v = self.fc2(v)
        v = F.relu(v)
        v = self.fc3(v)
        v = F.relu(v)
        v = self.fc4(v)
        v = F.relu(v)
        v = torch.cat([v, x], dim=1)
        v = self.fc5(v)
        v = F.relu(v)
        v = self.fc6(v)
        v = F.relu(v)
        v = self.fc7(v)
        v = F.relu(v)
        v = self.fc8(v)
        # add noise
        sigma = v[:, 0].clone()
        v = v[:, 1:].clone()    # no activation
        v = torch.cat([v, d], dim=1)
        v = self.add_fc1(v)
        v = F.relu(v)
        v = self.add_fc2(v)
        rgb = torch.sigmoid(v)
        noise = torch.tensor(np.random.normal(loc=0, scale=1, size=sigma.shape)).to(device=device, dtype=torch.float32)
        sigma = sigma + noise
        sigma = F.relu(sigma)
        return sigma, rgb
