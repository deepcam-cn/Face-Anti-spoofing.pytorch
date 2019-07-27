import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import random

def make_dataset(rgb_dir, depth_dir=None):
    items = []
    for file in os.listdir(rgb_dir):
        if file.endswith('.bmp') or file.endswith('.jpg') or file.endswith('.png'):
            if depth_dir is not None:
                depth_file = file[:-4] + '_depth.jpg'
                depth_file = os.path.join(depth_dir, depth_file)
                if os.path.exists(depth_file):
                    items.append((os.path.join(rgb_dir, file), depth_file))
            else:
                items.append((os.path.join(rgb_dir, file), ''))
    return items


class Dataset(data.Dataset):
    def __init__(self, mode, live_rgb, live_depth, fake_rgb, random_transform=None, target_transform=None):
        self.live_imgs = make_dataset(live_rgb, live_depth)
        self.fake_imgs = make_dataset(fake_rgb)
        self.live_len = len(self.live_imgs)
        self.fake_len = len(self.fake_imgs)
        self.mode = mode
        print(mode, ': live image size:', len(self.live_imgs))
        print(mode, ': fake image size:', len(self.fake_imgs))
        if len(self.live_imgs) == 0 or len(self.fake_imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.random_transform = random_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'train':
            if index < self.live_len:
                img_path, depth_path = self.live_imgs[index]
                img = Image.open(img_path).convert('RGB')
                depth = Image.open(depth_path).convert('L')
                if self.random_transform is not None:
                    img = self.random_transform(img)
                if self.target_transform is not None:
                    depth = self.target_transform(depth)
                return img, depth, 1
            else:
                img_path, depth_path = self.fake_imgs[index - self.live_len]
                img = Image.open(img_path).convert('RGB')
                if self.random_transform is not None:
                    img = self.random_transform(img)
                depth = torch.zeros(1, 32, 32)
                return img, depth, 0
        if self.mode == 'test':
            if index < self.live_len:
                img_path, _ = self.live_imgs[index]
                img = Image.open(img_path).convert('RGB')
                if self.random_transform is not None:
                    img = self.random_transform(img)
                return img, 1
            else:
                img_path, _ = self.fake_imgs[index - self.live_len]
                img = Image.open(img_path).convert('RGB')
                if self.random_transform is not None:
                    img = self.random_transform(img)
                return img, 0
        return


    def __len__(self):
        return self.live_len + self.fake_len
