import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None):
        
        self.mfcc = np.load('mfcc_feature.npy')
        
        with open(data_file, 'r') as file:
            self.data = []
            idx = 0
            for line in file:
                numbers = np.array(list(map(float, line.split())))  # Read and split numbers
                # self.data.append(np.hstack((numbers, self.mfcc[idx].reshape(-1))))
                self.data.append(numbers)
                # idx += 1

            feature_i = list()
            self.mean = list()
            self.std = list()
            for i in range(len(self.data[0])):
                feature_i.append(np.array([arr[i] for arr in self.data]))
                self.mean.append(np.mean(feature_i[i]))
                self.std.append(np.std(feature_i[i]))
            self.transform = transforms.Normalize(mean=self.mean, std=self.std)
            # self.transform = None
            print(self.std)
            for i in range(4):
                print(self.data[i])

        with open(labels_file, 'r') as file:
            self.labels = []
            for line in file:
                label = list(map(float, line.split()))
                label = label[2:4]
                self.labels.append(label)
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': torch.tensor(self.data[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.float) 
        }
        if self.transform:
            sample = {
            'data': torch.tensor((self.data[idx]-self.mean)/self.std, dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.float) 
            }
        return sample

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        loss_function = nn.L1Loss()  # Using Mean Squared Error loss
        loss_1 = loss_function(predicted[0], target[0])  # Loss for first output
        loss_2 = loss_function(predicted[1], target[1])  # Loss for second output
        total_loss = loss_1 + loss_2  # Combine the losses

        return total_loss