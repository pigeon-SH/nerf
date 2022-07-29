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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepVoxels():
    def __init__(self):
        self.root = "../datasets/DeepVoxels/synthetic_scenes/train/armchair"

        self.pose_fpaths = sorted(glob(os.path.join(self.root, "pose/*.txt")))
        self.depth_fpaths = sorted(glob(os.path.join(self.root, "depth/*.png")))
        self.img_fpaths = sorted(glob(os.path.join(self.root, "rgb/*.png")))
        self.total_img = len(self.img_fpaths)

        with open(os.path.join(self.root, "intrinsics.txt"), "r") as intrinsic_f:
            lines = intrinsic_f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    self.f, self.cx, self.cy, _ = map(float, line.split())
                    # K inverse formula: https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/
                    self.K_inv = np.zeros((3, 3))
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

    def __len__(self):
        return self.total_img * self.H * self.W
    
    def to_polar(self, xyz):
        x, y, z = xyz
        theta = np.arctan(y / x)
        phi = np.arctan(z / y)
        r = np.linalg.norm(xyz, 2)
        return theta, phi, r

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



    def get_rays(self, idx, pixel_x, pixel_y):        
        """return (camera pose o, viewing direction d, ground truth gt)
        """
        # https://github.com/vsitzmann/deepvoxels 
        # dataset "pose" files are matrix transformation that transform camera coord to world coord
        extrinsic_inv = np.genfromtxt(self.pose_fpaths[idx])
        extrinsic_inv = extrinsic_inv.reshape(4, 4)
        R_inv = extrinsic_inv[:3, :3]
        T_inv = extrinsic_inv[:3, 3]

        cam_pos = T_inv

        # get world coordinate vector of a pixel v1
        pixel = np.array([pixel_x, pixel_y, 1]) # [u, v]
        cam_coord = np.matmul(self.K_inv, pixel.T) # pixel's camera coord = pixel's cam direction since cam is at origin of cam coord
        cam_coord = np.append(cam_coord, 1)
        world_coord = np.matmul(extrinsic_inv, cam_coord)
        theta, phi, _ = self.to_polar(world_coord[:3])
        direction = np.array([theta, phi, 1])
        
        img = cv2.imread(self.img_fpaths[idx])
        gt = torch.tensor(img[pixel_y, pixel_x]).to(device=device)

        o_ndc, d_ndc = self.to_ndc(cam_pos, direction)
        o_ndc = torch.tensor(o_ndc).to(device=device, dtype=torch.float32)
        d_ndc = torch.tensor(d_ndc).to(device=device, dtype=torch.float32)
        
        return o_ndc, d_ndc, gt
    
    def cam_render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(100):
            extrinsic = np.genfromtxt(self.pose_fpaths[i])
            extrinsic = extrinsic.reshape(4, 4)
            
            R = extrinsic[:3, :3]
            T = extrinsic[:3, 3]
            cam_pos = -np.matmul(R, T)
            ax.scatter(*cam_pos)
        
        plt.show()

        
    def get_batch(self, samples):
        # randomly choose 4096 values from range(total_img * H * W)
        o_batch = []
        d_batch = []
        gt_batch = []
        for sample in samples:
            # sample = idx * (self.H*self.W) + y * (self.W) + x
            idx = sample // (self.H * self.W)
            s = sample - idx * (self.H * self.W)
            y = s // (self.W)
            s = s - y * (self.W)
            x = s
            o, d, gt = self.get_rays(idx, x, y)
            o_batch.append(o)
            d_batch.append(d)
            gt_batch.append(gt)
        o_batch = torch.stack(o_batch, dim=0)
        d_batch = torch.stack(d_batch, dim=0)
        gt_batch = torch.stack(gt_batch, dim=0)
        return o_batch, d_batch, gt_batch

class DataLoader():
    def __init__(self, dataset, batch_size=4096, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.samples)
            
    
    def get_batch(self, idx):
        return self.dataset.get_batch(self.samples[idx*self.batch_size : (idx+1)*self.batch_size])
    
    def __len__(self):
        return len(self.dataset) // self.batch_size