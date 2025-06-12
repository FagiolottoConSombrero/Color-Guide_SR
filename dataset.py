import os
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class AradDataset(Dataset):
    def __init__(self, x_dir, x_rgb, y_dir, transform_rgb=None, stereo=False):
        self.x_dir = x_dir
        self.x_rgb = x_rgb
        self.y_dir = y_dir
        self.file_list = sorted([
            f for f in os.listdir(x_dir)
            if f.endswith('.h5') and not f.startswith('._')
        ])
        self.transform_rgb = transform_rgb
        self.stereo = stereo

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_base = os.path.splitext(file_name)[0]  # rimuove .h5

        # Percorsi
        x_path = os.path.join(self.x_dir, file_name)
        y_path = os.path.join(self.y_dir, file_name)
        if self.stereo:
            rgb_path = os.path.join(self.x_rgb, file_base + '_rgb.tiff')
        else:
            rgb_path = os.path.join(self.x_rgb, file_base + '.jpg')

        # Carica i dati
        x = self._load_h5(x_path)
        y = self._load_h5(y_path)
        rgb = self._load_rgb(rgb_path)

        return x, rgb, y

    def _load_h5(self, path):
        with h5py.File(path, 'r') as f:
            data = f['data'][()]
            if self.stereo:
                data = torch.tensor(data, dtype=torch.float32)/ 255.0
            else:
                data = torch.tensor(data, dtype=torch.float32)
        return data

    def _load_rgb(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform_rgb:
            img = self.transform_rgb(img)
        else:
            img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0  # normalizza 0-1
            img = img.permute(2, 0, 1)  # da (H, W, C) a (C, H, W)
            if self.stereo:
                img = img[:, 4:268, 4:508]
            else:
                img = img[:, 1:481, 4:508]
        return img
