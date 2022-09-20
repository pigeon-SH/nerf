import os
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
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from nerf-pytorch
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
VAL_DIR = 'val0'

def render(net_coarse, net_fine, o_batch, d_batch, batch_size, is_noise):
    # hyperparameters
    near = 0.
    far = 1.
    coarse_num = 64
    fine_num = 128
    sample_num = coarse_num + fine_num

    # coarse sampling
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
    sigma_coarse, rgb_coarse = net_coarse(x_pe_coarse, d_pe_coarse, is_noise=is_noise)
    sigma_coarse = sigma_coarse.view((batch_size, coarse_num, 1))
    rgb_coarse = rgb_coarse.view((batch_size, coarse_num, 3))
    
    # compute volume rendering and get weights for fine sampling
    coarse_color, weights = utils.volume_rendering(t_coarse, sigma_coarse, rgb_coarse)
    weights += 1e-5 # prevent nans

    # fine sampling
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

    # concat t_coarse and t_fine
    t_fine = torch.cat([t_coarse, t_fine], dim=1)   # shape: (batch_size, sample_num, 1)
    t_fine = t_fine.sort(dim=1)[0]

    o_fine = torch.repeat_interleave(o_batch.view((batch_size, 1, 3)), repeats=sample_num, dim=1)    # shape: (batch_size, sample_num, 3)
    d_fine = torch.repeat_interleave(d_batch.view((batch_size, 1, 3)), repeats=sample_num, dim=1)    # shape: (batch_size, sample_num, 3)
    x_fine = o_fine + t_fine * d_fine

    # positional encoding for each fine samples
    x_pe_fine = utils.positional_encoding(x_fine, L=10)    # shape: (batch_size, sample_num, 60)
    d_pe_fine = utils.positional_encoding(d_fine, L=4)   # shape: (batch_size, sample_num, 24)
    
    # get density and color from model
    sigma_fine, rgb_fine = net_fine(x_pe_fine, d_pe_fine, is_noise=is_noise)
    sigma_fine = sigma_fine.view((batch_size, sample_num, 1))
    rgb_fine = rgb_fine.view((batch_size, sample_num, 3))

    # compute volume rendering
    fine_color, _ = utils.volume_rendering(t_fine, sigma_fine, rgb_fine)

    return coarse_color, fine_color


def train():
    if not os.path.exists(VAL_DIR):
        os.makedirs(VAL_DIR)
    # hyperparameters
    batch_size = 1024
    learning_rate = 5 * 1e-4
    min_learning_rate = 5 * 1e-5
    max_iter = 200000
    print_iter = 100
    val_iter = 500

    # dataset
    dataset = datasets.Lego(split="train")
    dataset_val = datasets.Lego(split="val")

    # model
    net_coarse = Network()
    net_coarse = net_coarse.to(device=device, dtype=torch.float32)
    net_fine = Network()
    net_fine = net_fine.to(device=device, dtype=torch.float32)

    # optimizer
    params = list(net_coarse.parameters())
    params += list(net_fine.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    losses = []
    loss_sum = 0

    # iteration
    iters = 0
    pbar = tqdm.tqdm(total=max_iter)
    while iters < max_iter:
        if iters % val_iter == 0:
            validate(net_coarse, net_fine, batch_size, dataset_val, 0, iters)

        # batch rays from loader
        idx = np.random.randint(dataset.img_num)        
        o_batch, d_batch, gt_batch = dataset.get_ray_batchs(idx, batch_size)
        o_batch = o_batch.to(device=device)
        d_batch = d_batch.to(device=device)
        gt_batch = gt_batch.to(device=device)

        coarse_color, fine_color = render(net_coarse, net_fine, o_batch, d_batch, batch_size, True)
        if not torch.all(fine_color>=0) or torch.all(fine_color==0):
            tqdm.tqdm.write("iter: {} fine_color neg or all_zero".format(iters))

        # compute loss
        optimizer.zero_grad()
        #loss = torch.sum(torch.square(torch.norm(coarse_color - gt_batch, p=2, dim=1)) + torch.square(torch.norm(fine_color - gt_batch, p=2, dim=1)))
        loss = torch.mean(torch.square(coarse_color - gt_batch)) + torch.mean(torch.square(fine_color - gt_batch))

        # backprop and optimizer step
        loss.backward()
        optimizer.step()
        #loss_sum += loss.item() / batch_size
        loss_sum += loss.item()

        iters += 1
        pbar.update(1)
        
        if iters >= max_iter:
            break

        if iters % print_iter == 0:
            torch.save(net_coarse, VAL_DIR + '/coarse_params.pt')
            torch.save(net_fine, VAL_DIR + '/fine_params.pt')
            loss_avg = loss_sum / print_iter
            losses.append(loss_avg)
            tqdm.tqdm.write("iter: {:7d}    loss: {:5.3f}".format(iters, loss_avg))
            loss_sum = 0

    pbar.close()
    plt.plot(losses)
    plt.savefig(VAL_DIR + '/loss_graph.png')

def validate(net_coarse, net_fine, batch_size, dataset_val, val_img_idx, iter):
    # model
    batch_size = 1024
    with torch.no_grad():
        o_batchs, d_batchs = dataset_val.get_rays_val(val_img_idx, batch_size)
        o_batchs, d_batchs = o_batchs.to(device=device), d_batchs.to(device=device)
        rgb = []
        for o_val, d_val in zip(o_batchs, d_batchs):
            fine_color, _ = render(net_coarse, net_fine, o_val, d_val, batch_size, False)
            rgb.append(fine_color)
        rgb = torch.stack(rgb, dim=0)
        rgb = rgb.view((dataset_val.W, dataset_val.H, 3))
        img = rgb.detach().cpu().numpy()
        img = to8b(img)
        plt.imsave(VAL_DIR + "/iter{}.png".format(iter), img)


def test():
    dataset_val = datasets.Lego(split="test")

    # model
    net_coarse = torch.load(VAL_DIR + '/coarse_params.pt')
    net_coarse = net_coarse.to(device=device, dtype=torch.float32)
    net_fine = torch.load(VAL_DIR + '/fine_params.pt')
    net_fine = net_fine.to(device=device, dtype=torch.float32)
    batch_size = 1024
    with torch.no_grad():
        rgbs = []
        for idx in tqdm.tqdm(range(dataset_val.img_num)):
            o_batchs, d_batchs = dataset_val.get_rays_val(idx, batch_size)
            o_batchs, d_batchs = o_batchs.to(device=device), d_batchs.to(device=device)
            rgb = []
            for o_val, d_val in zip(o_batchs, d_batchs):
                fine_color, _ = render(net_coarse, net_fine, o_val, d_val, batch_size)
                rgb.append(fine_color)
            rgb = torch.stack(rgb, dim=0).view((dataset_val.W, dataset_val.H, 3))
            rgb = rgb.detach().cpu().numpy()
            rgbs.append(rgb)
        movie_path = os.path.join(VAL_DIR, "val_rgb.mp4")
        imageio.mimwrite(movie_path, to8b(rgbs), fps=30, quality=8)


train()
test()