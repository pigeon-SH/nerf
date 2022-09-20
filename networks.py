import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 63: positional encoding 2*L=20 for each x, y, z and concat original value 3
        self.fc1 = nn.Linear(63, 256)   
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256 + 63, 256) # skip connection
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)  
        self.sigma_layer = nn.Linear(256, 1)    # sigma output
        self.feature_layer = nn.Linear(256, 256)
        # 27: positional encoding 2*L=8 for each direction(x, y, z) and concat original value 3
        self.add_fc1 = nn.Linear(256 + 27, 128) 
        self.add_fc2 = nn.Linear(128, 3)    # rgb output
    
    def forward(self, x, d, is_noise):
        """ x: result of positional encoding of location, shape=(batch_size, sample_num, 63)
            d: result of positional encoding of viewing direction, shape=(batch_size, sample_num, 27)
            is_noise: if train, apply noise else(test or validate), do not apply noise
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
        v = self.fc5(v)
        v = F.relu(v)
        v = torch.cat([v, x], dim=1)
        v = self.fc6(v)
        v = F.relu(v)
        v = self.fc7(v)
        v = F.relu(v)
        v = self.fc8(v)
        # add noise
        sigma = self.sigma_layer(v)
        if is_noise:
            noise = torch.tensor(np.random.normal(loc=0, scale=1, size=sigma.shape)).to(device=device, dtype=torch.float32)
            sigma = sigma + noise
        sigma = F.relu(sigma)

        v = self.feature_layer(v)
        v = torch.cat([v, d], dim=1)
        v = self.add_fc1(v)
        v = F.relu(v)
        v = self.add_fc2(v)
        rgb = torch.sigmoid(v)
        return sigma, rgb
