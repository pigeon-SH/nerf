import numpy as np
import os
from glob import glob
import cv2
from skimage import transform
import torch
import random

import utils
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

import sys

device = torch.device("cpu")

class DeepVoxels():
    def __init__(self, is_train, is_ndc):
        if is_train:
            self.root = "../datasets/DeepVoxels/synthetic_scenes/train/armchair"
        else:
            self.root = "../datasets/DeepVoxels/synthetic_scenes/validation/armchair"
        self.is_ndc = is_ndc

        self.pose_fpaths = sorted(glob(os.path.join(self.root, "pose/*.txt")))
        self.depth_fpaths = sorted(glob(os.path.join(self.root, "depth/*.png")))
        self.img_fpaths = sorted(glob(os.path.join(self.root, "rgb/*.png")))
        self.img_num = len(self.img_fpaths)

        with open(os.path.join(self.root, "intrinsics.txt"), "r") as intrinsic_f:
            lines = intrinsic_f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    self.f, self.cx, self.cy, _ = map(float, line.split())
                    # K inverse formula: https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/
                    self.K_inv = torch.zeros((3, 3), device=device, dtype=torch.float32)
                    self.K_inv[0, 0] = 1
                    self.K_inv[1, 1] = 1
                    self.K_inv[0, 2] = -self.cx
                    self.K_inv[1, 2] = -self.cy
                    self.K_inv[2, 2] = self.f
                    self.K_inv /= self.f
                elif i == 2:
                    self.n = float(line.strip())
                    if self.n == 0:
                        self.n == np.sqrt(3) / 2
                elif i == len(lines) - 1:
                    #self.H, self.W = map(int, map(float, line.split()))
                    self.H, self.W = 512, 512
        
        # 
        self.o_s = []
        self.d_s = []
        self.gt_s = []
        for idx in range(self.img_num):
            extrinsic_inv = np.genfromtxt(self.pose_fpaths[idx])
            extrinsic_inv = torch.tensor(extrinsic_inv.reshape(4, 4), device=device, dtype=torch.float32)
            img = cv2.imread(self.img_fpaths[idx])  # shape: (H, W, 3)
            gt = torch.tensor(img, device=device).permute((1, 0, 2)) / 255.    # shape: (W, H, 3)
            self.gt_s.append(gt)

            T_inv = extrinsic_inv[:3, 3]
            o = T_inv
            self.o_s.append(o)
        
            xs = torch.arange(0, self.W).to(device=device, dtype=torch.float32)
            ys = torch.arange(0, self.H).to(device=device, dtype=torch.float32)
            xs, ys = torch.meshgrid(ys, xs)   # default indexing='ij' for pytorch version 1.7
            pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).view((self.W, self.H, 3, 1))    # shape: (W, H, 3, 1)
            d_cam = torch.matmul(self.K_inv, pixels)    # shape: (W, H, 3, 1)
            d_cam = torch.cat([d_cam[:, :, 0:1], -d_cam[:, :, 1:2], -torch.ones_like(d_cam[:, :, 0:1])], dim=2) # shape: (W, H, 3, 1)
            d_world = torch.matmul(extrinsic_inv[:3, :3], d_cam).view((self.W, self.H, 3))  # shape: (W, H, 3)
            self.d_s.append(d_world)
                        
        self.o_s = torch.stack(self.o_s, dim=0) # shape: (img_num, 3)
        self.d_s = torch.stack(self.d_s, dim=0).view((self.img_num, self.W, self.H, 3)) # shape: (img_num, W, H, 3)
        self.gt_s = torch.stack(self.gt_s, dim=0).view((self.img_num, self.W, self.H, 3))   # shape: (img_num, W, H, 3)

        self.d_s = self.d_s / torch.norm(self.d_s, p=2, dim=-1, keepdim=True)

    def __len__(self):
        return self.img_num

    def to_polar(self, xyz):
        """xyz shape: (W, H, 3)
        """
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]
        theta = torch.arctan(y / x)
        phi = torch.arctan(z / torch.sqrt(torch.square(x) + torch.square(y)))
        return theta, phi   # shape: (W, H)

    def to_ndc(self, o, d):
        o_prime = np.zeros(o.size)
        o_prime[0] = -self.f * 2 / self.W * o[0] / o[2]
        o_prime[1] = -self.f * 2 / self.H * o[1] / o[2]
        o_prime[2] = 1 + 2 * self.n / o[2]

        d_prime = np.zeros(d.size)
        d_prime[0] = -self.f * 2 / self.W * (d[0] / d[2] - o[0] / o[2])
        d_prime[1] = -self.f * 2 / self.H * (d[1] / d[2] - o[1] / o[2])
        d_prime[2] = -2 * self.n / o[2]

        return o_prime, d_prime
    
    def get_rays_val(self, idx, batch_size):
        d_f = torch.flatten(self.d_s[idx], 0, 1)    # shape: (W*H, 3)
        i = 0
        d_batchs = []
        while (i+1)*batch_size < self.W * self.H:
            start = i * batch_size
            end = (i+1) * batch_size 
            d = d_f[start:end, :]
            d_batchs.append(d)
            i += 1
        d = d_f[start:end, :]
        d_batchs.append(d)
        d_batchs = torch.stack(d_batchs, 0) # shape: (num, batch_size, 3) where (num*batch_size)=W*H

        o_batch = self.o_s[idx:idx+1]   # shape: (1, 3)
        o_batch = torch.repeat_interleave(o_batch.view((1, 1, 3)), batch_size, dim=1)   # shape: (1, batch_size, 3)
        o_batchs = torch.repeat_interleave(o_batch, len(d_batchs), dim=0) # shape: (num, batch_size, 3) where (num*batch_size)=W*H

        return o_batchs, d_batchs
    
    def get_ray_batchs(self, idx, batch_size):
        o_batch = self.o_s[idx:idx+1]   # shape: (1, 3)
        o_batch = torch.repeat_interleave(o_batch, batch_size, dim=0)   # shape: (batch_size, 3)
        pixel_idxs = torch.randint(0, self.W * self.H, (batch_size,))
        d_f = torch.flatten(self.d_s[idx], 0, 1)    # shape: (W*H, 3)
        d_batch = d_f[pixel_idxs, :]
        gt_f = torch.flatten(self.gt_s[idx], 0, 1)    # shape: (W*H, 3)
        gt_batch = gt_f[pixel_idxs, :]
        return o_batch, d_batch, gt_batch




