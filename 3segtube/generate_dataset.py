import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from torchvision import transforms
from sklearn.decomposition import PCA

class CustomDataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None):
        
        self.data = list()
        self.mfcc = np.load('mfcc_feature.npy')
        self.td = np.load('td_feature.npy')
        self.formants = np.load('formant_feature.npy')

        for i in range(len(self.mfcc)):
            self.data.append(np.hstack((self.formants[i].reshape(-1), self.mfcc[i].reshape(-1), self.td[i].reshape(-1))))
        '''
        with open(data_file, 'r') as file:
            self.data = []
            idx = 0
            for line in file:
                numbers = np.array(list(map(float, line.split())))  # Read and split numbers
                self.data.append(np.hstack((numbers, self.mfcc[idx].reshape(-1), self.td[idx].reshape(-1))))
                # self.data.append(numbers)
                idx += 1
        '''

        feature_i = list()
        self.mean = list()
        self.std = list()
        for i in range(len(self.data[0])):
            feature_i.append(np.array([arr[i] for arr in self.data]))
            self.mean.append(np.mean(feature_i[i]))
            self.std.append(np.std(feature_i[i]))
        
        for i, features in enumerate(self.data):
            self.data[i] = (features - self.mean) / self.std
        self.transform = None
        data_array = np.vstack(self.data)
        
        # Apply PCA here to reduce the dimension
        num_components = 14  # Choose the number of components
        pca = PCA(n_components=num_components)
        self.data = pca.fit_transform(data_array)
        print(self.data.shape)

        with open(labels_file, 'r') as file:
            self.labels = []
            for line in file:
                label = list(map(float, line.split()))
                label = label[2:6]
                self.labels.append(label)
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': torch.tensor(self.data[idx,:], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.float) 
        }
        if self.transform:
            sample = {
            'data': torch.tensor(self.data[idx,:], dtype=torch.float),
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
        loss_3 = loss_function(predicted[2], target[2])  # Loss for second output
        loss_4 = loss_function(predicted[3], target[3])  # Loss for second output


        total_loss = loss_1 + loss_2 + loss_3 + loss_4# Combine the losses

        return total_loss