import numpy as np
import torch
import random
import os
from itertools import repeat
from torch.utils.data import Dataset


class ReconDataset(Dataset):

    def __init__(self, year, split="train", stride=1, numframes=16):
        super().__init__()
        self.root_path = f"../../../data/MESO-5km/{year}-zscore"
        self.dIndex = np.load(f"{self.root_path}/index-p-{year}.npy")
        self.data_length = self.dIndex.shape[0]
        self.stride = stride
        self.vname = ('GPH100', 'GPH500', 'GPH850', 'SLP', 'RH500', 'RH850',
                'T2', 'T500', 'T850', 'U10', 'U500', 'U850', 'V10', 'V500', 'V850', 'RAIN')


    def __getitem__(self, index):
        index *= self.stride
        dIn = self.dIndex[index]

        args = zip(repeat(dIn), range(0, 16))
        data = list(map(self, args))
        data = np.array(data, dtype=np.float32)
        return data


    def __call__(self, pdata):
        datetime, idx = pdata
        data_path = os.path.join(self.root_path, self.vname[idx])
        return np.load(f"{data_path}/{datetime}.npy")


    def __len__(self):
        num_of_idx = self.data_length // self.stride
        return num_of_idx



class ProbDataset(Dataset):

    def __init__(self, year, split="train", stride=1, numframes=16):
        super().__init__()
        self.root_path = f"../../../data/MESO-5km/{year}-zscore"
        if split == "train":
            if year == 2023:
                self.dIndex = np.load(f"{self.root_path}/index-{year}-all.npy")
            else:
                self.dIndex = np.load(f"{self.root_path}/index-{year}.npy")
        self.data_length = self.dIndex.shape[0]
        self.stride = stride
        self.vname = ('GPH100', 'GPH500', 'GPH850', 'SLP', 'RH500', 'RH850',
                'T2', 'T500', 'T850', 'U10', 'U500', 'U850', 'V10', 'V500', 'V850', 'RAIN', 'QPE', 'QPE')
        self.mask = np.load("../../../data/qpe-korea-map.npy")


    def __getitem__(self, index):
        index *= self.stride
        dIn = self.dIndex[index]
        args = zip(repeat(dIn), range(0, 18))
        data = list(map(self, args))
        return np.array(data[:16], dtype=np.float32), data[16], data[17]


    def __call__(self, pdata, alpha=.0):
        datetime, idx = pdata
        data_path = os.path.join(self.root_path, self.vname[idx])
        data = np.load(f"{data_path}/{datetime}.npy")
        inds = np.where(self.korea_index==False)

        """ One-hot labeling """
        if idx == 16:
            data *= self.mask
            data = np.where(data >= 10., 2, np.where(data >= 0.1, 1, 0))

        """ Probabilistic density labeling """
        if idx == 17:
            data *= self.korea_index
            data0 = np.where(data < 0, (1-alpha)+alpha/3, 
                    np.where(data == 0., (1-alpha)+alpha/3, np.where(data < 0.1, (1-alpha)*(0.1-data)/0.1+alpha/3, 0+alpha/3)))
            data1 = np.where(data <= 0., alpha/3, np.where(data <= 0.1, (1-alpha)*np.round(1-(0.1-data)/0.1, 2)+alpha/3,
                    np.where(data < 10, (1-alpha)*np.round((10-data)/9.9, 2)+alpha/3, 0+alpha/3)))
            data2 = np.where(data <= 0.1, alpha/3, np.where(data <= 10., (1-alpha)*np.round(1-(10-data)/9.9, 2)+alpha/3, (1-alpha)+alpha/3))
            data = np.stack([data0, data1, data2])
        return data


    def __len__(self):
        num_of_idx = self.data_length // self.stride
        return num_of_idx




