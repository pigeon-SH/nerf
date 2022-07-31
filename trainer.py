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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # hyperparameters
    batch_size = 2048
    coarse_num = 64
    fine_num = 128
    learning_rate = 5 * 1e-4
    weight_decay = 5 * 1e-5
    max_iter = 100000
    val_iter = 100

    # dataset
    dataset = datasets.DeepVoxels()
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        for idx, sample in enumerate(dataloader):
            o_batch = sample[0]["o"]
            d_batch = sample[0]["d"]
            gt_batch = sample[1]

            # coarse sampling: because of ndc, we can sampling t from 0 to 1
            x_batch = []
            for i in range(batch_size):
                coarse_samples = torch.tensor([random.uniform(c*(1/coarse_num), (c+1)*(1/coarse_num)) for c in range(coarse_num)]).to(device=device, dtype=torch.float32)
                x = torch.stack([o_batch[i] + t * d_batch[i] for t in coarse_samples], dim=0)   # shape: (coarse_num, 3)  
                x_batch.append(x)
            x_batch = torch.stack(x_batch, dim=0)   # shape: (batch_size, coarse_num, 3)

            # positional encoding for each coarse samples
            x_pe = utils.positional_encoding(x_batch, L=10) # shape: (batch_size, coarse_num, 60)
            d = d_batch.view((batch_size, 1, 3))    # shape: (batch_size, 3) => (batch_size, 1, 3)
            d = torch.repeat_interleave(d, repeats=coarse_num, dim=1)
            d_pe = utils.positional_encoding(d, L=4)  # shape: (batch_size, coarse_num, 24)

            # get density and color from model
            x_pe_flatten = torch.flatten(x_pe, 0, 1)    # shape: (batch_size*coarse_num, 60)
            d_pe_flatten = torch.flatten(d_pe, 0, 1)    # shape: (batch_size*coarse_num, 24)
            sigma, rgb = model(x_pe_flatten, d_pe_flatten)
            sigma = sigma.view((batch_size, coarse_num, 1))
            rgb = rgb.view((batch_size, coarse_num, 3))
            
            # compute volume rendering and get weights for fine sampling
            coarse_color_batch = []
            weight_batch = []
            for i in range(batch_size):
                color, weights = utils.volume_rendering(x_batch[i], sigma[i], rgb[i])
                coarse_color_batch.append(color)
                weight_batch.append(weights)
            coarse_color_batch = torch.stack(coarse_color_batch, dim=0)
            weight_batch = torch.stack(weight_batch, dim=0)

            # fine sampling
            total_batch = torch.sum(weight_batch, dim=1, keepdim=True)
            pdf_batch = weight_batch / total_batch
            cdf_batch = torch.cumsum(pdf_batch, dim=1)
            cdf_batch = torch.cat([torch.zeros_like(cdf_batch[:, 0:1]), cdf_batch], dim=1)  # for interpolate, minimum value of cdf need to be 0
            cdf_batch = cdf_batch.view(cdf_batch.shape[:2])
            x = torch.linspace(0, 1, coarse_num + 1)    # +1: for interpolate, number of x values should be same with number of y values
            cdf_inv_batch = [scipy.interpolate.interp1d(cdf.detach().cpu().numpy(), x) for cdf in cdf_batch]
            fine_x_batch = []
            for i in range(batch_size):
                uniform_samples = np.random.uniform(0, 1, size=(fine_num))
                fine_samples = cdf_inv_batch[i](uniform_samples)
                x = torch.stack([o_batch[i] + t * d_batch[i] for t in fine_samples], dim=0)    
                fine_x_batch.append(x)
            fine_x_batch = torch.stack(fine_x_batch, dim=0) # shape: (batch_size, fine_num, 3)

            # positional encoding for each fine samples
            fine_x_pe = utils.positional_encoding(fine_x_batch, L=10)    # shape: (batch_size, fine_num, 60)
            d = d_batch.view((batch_size, 1, 3))    # shape: (batch_size, 3) => (batch_size, 1, 3)
            d = torch.repeat_interleave(d, repeats=fine_num, dim=1)
            fine_d_pe = utils.positional_encoding(d, L=4)   # shape: (batch_size, fine_num, 24)
            
            x_pe = torch.cat([x_pe, fine_x_pe], dim=1)
            d_pe = torch.cat([d_pe, fine_d_pe], dim=1)
            x_pe_flatten = torch.flatten(x_pe, 0, 1)
            d_pe_flatten = torch.flatten(d_pe, 0, 1)
            
            # get density and color from model
            sigma, rgb = model(x_pe_flatten, d_pe_flatten)
            sigma = sigma.view((batch_size, fine_num + coarse_num, 1))
            rgb = rgb.view((batch_size, fine_num + coarse_num, 3))

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

            # compute loss
            optimizer.zero_grad()
            loss = torch.sum(torch.square(torch.norm(coarse_color_batch - gt_batch, p=2, dim=1)) + torch.square(torch.norm(fine_color_batch - gt_batch, p=2, dim=1)))
            
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