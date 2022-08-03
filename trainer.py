import torch
import random
import numpy as np

from networks import Network
import datasets
import utils
import sys

import scipy
import tqdm
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    before_t = time.time()
    # hyperparameters
    batch_size = 1024
    coarse_num = 64
    fine_num = 128
    learning_rate = 5 * 1e-4
    weight_decay = 5 * 1e-5
    max_iter = 200000
    val_iter = 100

    near = 0.
    far = 1.

    # dataset
    dataset = datasets.DeepVoxels(is_ndc=False)
    # dataloader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = Network()
    model = model.to(device=device, dtype=torch.float32)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    losses = []

    # iteration
    iter = 0
    pbar = tqdm.tqdm(total=max_iter)
    while iter < max_iter:
        # batch rays from loader
        #for idx, sample in enumerate(dataloader):
        for idx in range(batch_size):
            sample = dataset.get_rays(batch_size)
            o_batch = sample[0]["o"]
            d_batch = sample[0]["d"]
            gt_batch = sample[1]

            # coarse sampling
            with torch.no_grad():
                low = torch.linspace(near, far, coarse_num + 1, device=device, dtype=torch.float32)[:-1]
                randoms = torch.rand((batch_size, coarse_num), device=device, dtype=torch.float32)
                t_coarse = low + (far - near) / coarse_num * randoms # shape: (batch_size, coarse_num)
                t_coarse = t_coarse.view((batch_size, coarse_num, 1)) # shape: (batch_size, coarse_num, 1)
                o_coarse = torch.repeat_interleave(o_batch.view((batch_size, 1, 3)), repeats=coarse_num, dim=1)    # shape: (batch_size, coarse_num, 3)
                d_coarse = torch.repeat_interleave(d_batch.view((batch_size, 1, 3)), repeats=coarse_num, dim=1)    # shape: (batch_size, coarse_num, 3)
                x_coarse = o_coarse + t_coarse * d_coarse   # shape: (batch_size, coarse_num, 3)

                # positional encoding for each coarse samples
                x_pe_coarse = utils.positional_encoding(x_coarse, L=10) # shape: (batch_size, coarse_num, 60)
                d_pe_coarse = utils.positional_encoding(d_coarse, L=4)  # shape: (batch_size, coarse_num, 24)

            # get density and color from model
            sigma_coarse, rgb_coarse = model(x_pe_coarse, d_pe_coarse)
            sigma_coarse = sigma_coarse.view((batch_size, coarse_num, 1))
            rgb_coarse = rgb_coarse.view((batch_size, coarse_num, 3))
            
            # compute volume rendering and get weights for fine sampling
            coarse_color, weights = utils.volume_rendering(t_coarse, sigma_coarse, rgb_coarse)
            weights += 1e-5 # prevent nans

            # fine sampling
            with torch.no_grad():
                total = torch.sum(weights, dim=1, keepdim=True)
                pdf_batch = weights / total
                cdf_batch = torch.cumsum(pdf_batch, dim=1)
                cdf_batch = torch.cat([torch.zeros_like(cdf_batch[:, 0:1]), cdf_batch], dim=1)  # for interpolate, minimum value of cdf need to be 0
                cdf_batch = cdf_batch.view(cdf_batch.shape[:2]) # shape: (batch_size, coarse_num + 1)
                x = torch.linspace(near, far, coarse_num + 1, device=device, dtype=torch.float32)    # +1: for interpolate, number of x values should be same with number of y values
                uniform_samples = torch.rand((batch_size, fine_num), device=device, dtype=torch.float32)
            
                indices = torch.searchsorted(cdf_batch, uniform_samples, right=True)    # shape: (batch_size, fine_num)
                left = indices - 1
                right = indices
                left[left < 0] = 0
                right[right == coarse_num + 1] = coarse_num - 1
                # invert cdf with linear interpolation https://en.wikipedia.org/wiki/Linear_interpolation
                y0 = x[left]
                y1 = x[right]
                left_oh = torch.nn.functional.one_hot(left, coarse_num + 1)
                right_oh = torch.nn.functional.one_hot(right, coarse_num + 1)
                cdf_batch = torch.repeat_interleave(cdf_batch.view((batch_size, 1, coarse_num + 1)), repeats=fine_num, dim=1)   # shape: (batch_size, fine_num, coarse_num + 1)
                x0 = torch.sum(cdf_batch * left_oh, dim=-1)
                x1 = torch.sum(cdf_batch * right_oh, dim=-1)
                t_fine = y0 * (x1 - uniform_samples) + y1 * (uniform_samples - x0) / (x1 - x0)  # shape: (batch_size, fine_num)
                t_fine = t_fine.view((batch_size, fine_num, 1))

                o_fine = torch.repeat_interleave(o_batch.view((batch_size, 1, 3)), repeats=fine_num, dim=1)    # shape: (batch_size, fine_num, 3)
                d_fine = torch.repeat_interleave(d_batch.view((batch_size, 1, 3)), repeats=fine_num, dim=1)    # shape: (batch_size, fine_num, 3)
                x_fine = o_fine + t_fine * d_fine

            # positional encoding for each fine samples
            x_pe_fine = utils.positional_encoding(x_fine, L=10)    # shape: (batch_size, fine_num, 60)
            d_pe_fine = utils.positional_encoding(d_fine, L=4)   # shape: (batch_size, fine_num, 24)
            
            x_pe_fine = torch.cat([x_pe_coarse, x_pe_fine], dim=1)
            d_pe_fine = torch.cat([d_pe_coarse, d_pe_fine], dim=1)
            
            # get density and color from model
            sigma_fine, rgb_fine = model(x_pe_fine, d_pe_fine)
            sigma_fine = sigma_fine.view((batch_size, fine_num + coarse_num, 1))
            rgb_fine = rgb_fine.view((batch_size, fine_num + coarse_num, 3))

            # compute volume rendering
            t_fine = torch.cat([t_coarse, t_fine], dim=1)
            fine_color, _ = utils.volume_rendering(t_fine, sigma_fine, rgb_fine)

            # compute loss
            optimizer.zero_grad()
            loss = torch.sum(torch.square(torch.norm(coarse_color - gt_batch, p=2, dim=1)) + torch.square(torch.norm(fine_color - gt_batch, p=2, dim=1)))
            
            # backprop and optimizer step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            iter += 1
            pbar.update(1)
            if iter >= max_iter:
                break

            if iter % val_iter == 0:
                torch.save(model, 'params.pt')
                # validate model per some iterations
                pass

    pbar.close()
    plt.plot(losses)
    plt.savefig('loss_graph.png')
    plt.show()


train()