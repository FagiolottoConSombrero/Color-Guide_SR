from PIL import Image
import numpy as np
import torch
import os
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader

t_data_path = '/home/matteo/Documents/arad1k/h5/train'
v_data_path = '/home/matteo/Documents/arad1k/h5/val'

train_x = os.path.join(t_data_path, 'train_arad1k_x4')
train_x_rgb = os.path.join(t_data_path, 'Train_RGB')
train_y = os.path.join(t_data_path, 'train_arad1k_original')
val_x = os.path.join(v_data_path, 'val_arad1k_x4')
val_x_rgb = os.path.join(t_data_path, 'Train_RGB')
val_y = os.path.join(v_data_path, 'val_arad1k_original')

print("===> Loading data")
train_set = AradDataset(train_x, train_x_rgb, train_y)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

valid_set = AradDataset(val_x, val_x_rgb, val_y)
valid_loader = DataLoader(valid_set, batch_size=8, shuffle=False)

for x, rgb, y in train_loader:
    print(f"x shape: {x.shape}")    # hyperspectral low-res
    print(f"rgb shape: {rgb.shape}")  # immagine RGB
    print(f"y shape: {y.shape}")    # hyperspectral high-res
    break  # solo il primo batch

