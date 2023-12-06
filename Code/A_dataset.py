
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch

class BEIJING18_DATASET(Dataset):
    def __init__(self, configs):
        super(BEIJING18_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/Bejing18_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/mask/beijing18/beijing18_"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()
        return data_res, mask_res, observed_tp, mask_gt

class BEIJING18_DATASET_TEMP(Dataset):
    def __init__(self, configs):
        super(BEIJING18_DATASET_TEMP, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/Bejing18_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/mask/beijing18/beijing18_"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        large_acc = np.loadtxt("Data/mask/beijing18/larger_acc.csv", delimiter=",")
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0

        for i in range(self.mask_gt.shape[0]):
            for j in range(self.mask_gt.shape[1]):
                if self.mask_gt[i,j] == 1 and large_acc[i,j] == 0:
                        if self.mask[i,j] == 0:
                            self.mask_gt[i,j] = 0

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()
        return data_res, mask_res, observed_tp, mask_gt

class URBANTRAFFIC_DATASET(Dataset):
    def __init__(self, configs):
        super(URBANTRAFFIC_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/UrbanTraffic_norm.csv", delimiter=",")
        self.mask = np.loadtxt("Data/mask/urbantraffic/urbantraffic_"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()
        return data_res, mask_res, observed_tp, mask_gt

class PHYSIONET12_DATASET(Dataset):
    def __init__(self, configs):
        super(PHYSIONET12_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/PhysioNet12_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/mask/physionet12/physionet12_"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()
        return data_res, mask_res, observed_tp, mask_gt

def get_physionet12_dataset(configs):
    dataset = PHYSIONET12_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

def get_beijing18_dataset(configs):
    dataset = BEIJING18_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

def get_urbantraffic_dataset(configs):
    dataset = URBANTRAFFIC_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader


def get_dataset(configs):
    if configs.dataset == "beijing18":
        return get_beijing18_dataset(configs)
    if configs.dataset == "physionet12":
        return get_physionet12_dataset(configs)
    if configs.dataset == "urbantraffic":
        return get_urbantraffic_dataset(configs)