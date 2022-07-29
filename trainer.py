import torch
import random
import numpy as np

from networks import Network
import datasets
import utils
import sys

import scipy
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # hyperparameters
    batch_size = 4096
    coarse_num = 64
    fine_num = 128
    learning_rate = 5 * 1e-4
    weight_decay = 5 * 1e-5
    max_iter = 300000
    val_iter = max_iter // 10

    # dataset
    dataset = datasets.DeepVoxels()
    # dataloader
    #dataloader = datasets.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = datasets.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = Network()
    model = model.to(device=device, dtype=torch.float32)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # iteration
    for iter in tqdm.tqdm(range(max_iter)):
        # batch rays from loader
        for idx in tqdm.tqdm(range(len(dataloader))):
            o_batch, d_batch, gt_batch = dataloader.get_batch(idx)
            print("get batch")

            # coarse sampling: because of ndc, we can sampling t' from 0 to 1
            x_batch = []
            for i in range(batch_size):
                coarse_samples = torch.tensor([random.uniform(c*(1/coarse_num), (c+1)*(1/coarse_num)) for c in range(coarse_num)]).to(device=device, dtype=torch.float32)
                x = torch.stack([o_batch[i] + t * d_batch[i] for t in coarse_samples], dim=0)    
                x_batch.append(x)
            x_batch = torch.stack(x_batch, dim=0)   # shape: (batch_size, coarse_num, 3)
            print("coarse sampling")

            # positional encoding for each coarse samples
            x_pe = torch.stack([utils.positional_encoding(x, L=10) for batch in x_batch for x in batch], dim=0).to(dtype=torch.float32)   # shape: (batch_size*coarse_num, 60)
            d_pe = torch.stack([utils.positional_encoding(d, L=4) for d in d_batch for _ in range(coarse_num)], dim=0).to(dtype=torch.float32)    # shape: (batch_size*coarse_num, 24)
            print("positional encoding")

            # get density and color from model
            sigma, rgb = model(x_pe, d_pe)
            sigma = sigma.view((batch_size, coarse_num, 1))
            rgb = rgb.view((batch_size, coarse_num, 3))
            print("model")
            
            # compute volume rendering and get weights for fine sampling
            coarse_color_batch = []
            weight_batch = []
            for i in range(batch_size):
                color, weights = utils.volume_rendering(x_batch[i], sigma[i], rgb[i])
                coarse_color_batch.append(color)
                weight_batch.append(weights)
            coarse_color_batch = torch.stack(coarse_color_batch, dim=0)
            weight_batch = torch.stack(weight_batch, dim=0)
            print("volume rendering")

            # fine sampling
            total_batch = torch.sum(weight_batch, dim=1, keepdim=True)
            pdf_batch = weight_batch / total_batch
            cdf_batch = torch.cumsum(pdf_batch, dim=1)
            cdf_batch = torch.cat([torch.zeros_like(cdf_batch[:, 0:1]), cdf_batch], dim=1)  # for interpolate, minimum value of cdf need to be 0
            x = torch.linspace(0, 1, coarse_num + 1)    # +1: for interpolate, number of x values should be same with number of y values
            cdf_inv_batch = [scipy.interpolate.interp1d(cdf, x) for cdf in cdf_batch]
            fine_x_batch = []
            for i in range(batch_size):
                uniform_samples = np.random.uniform(0, 1, size=(fine_num))
                fine_samples = cdf_inv_batch[i](uniform_samples)
                x = torch.stack([o_batch[i] + t * d_batch[i] for t in fine_samples], dim=0)    
                fine_x_batch.append(x)
            fine_x_batch = torch.stack(fine_x_batch, dim=0) # shape: (batch_size, fine_num, 3)
            print("fine sampling")

            # positional encoding for each fine samples
            fine_x_pe = torch.stack([utils.positional_encoding(x, L=10) for batch in fine_x_batch for x in batch], dim=0).to(dtype=torch.float32)   # shape: (batch_size*coarse_num, 60)
            fine_d_pe = torch.stack([utils.positional_encoding(d, L=4) for d in d_batch for _ in range(fine_num)], dim=0).to(dtype=torch.float32)    # shape: (batch_size*coarse_num, 24)
            
            x_pe = x_pe.view((batch_size, coarse_num, 60))
            d_pe = d_pe.view((batch_size, coarse_num, 24))
            x_pe = torch.cat([x_pe, fine_x_pe], dim=1)
            d_pe = torch.cat([d_pe, fine_d_pe], dim=1)
            x_pe = torch.flatten(x_pe, 0, 1)
            d_pe = torch.flatten(d_pe, 0, 1)
            print("fine positional encoding")
            
            # get density and color from model
            sigma, rgb = model(x_pe, d_pe)
            sigma = sigma.view((batch_size, fine_num + coarse_num, 1))
            rgb = rgb.view((batch_size, fine_num + coarse_num, 3))
            print("fine model")

            # compute volume rendering
            x_batch = torch.cat([x_batch, fine_x_batch], dim=1)
            fine_color_batch = []
            weight_batch = []
            for i in range(batch_size):
                color, weights = utils.volume_rendering(x_batch[i], sigma[i], rgb[i])
                fine_color_batch.append(color)
                weight_batch.append(weights)
            fine_color_batch = torch.stack(fine_color_batch, dim=0)
            weight_batch = torch.stack(weight_batch, dim=0)
            print("fine volume rendering")

            # compute loss
            optimizer.zero_grad()
            loss = torch.sum(torch.square(torch.norm(coarse_color_batch - gt_batch, p=2, dim=1)) + torch.square(torch.norm(fine_color_batch - gt_batch, p=2, dim=1)))
            print("loss")
            
            # backprop and optimizer step
            loss.backward()
            optimizer.step()
        # validate model per some iterations
        pass

train()