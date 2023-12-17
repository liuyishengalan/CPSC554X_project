import torch
from torch import nn
from generate_dataset import CustomDataset, CustomLoss
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# use cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3)
        #self.fc1 = nn.Linear(128*4 , 64)
        self.fc1 = nn.Linear(512 * 6, 128)  # Adjusted fully connected layer input size
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, x.size(0))
        # print(x.shape)
        #x = self.fc1(x)
        # print(x.shape)
        #x = nn.functional.relu(x)
        x = nn.functional.relu(self.fc1(x))
        #x = nn.functional.relu(self.fc2(x))
        # x = nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # print(x.shape)
        #x = self.dropout(x)
        # print(x.shape)
        #x = self.fc2(x)
        return x
  
if __name__ == '__main__':

    data_file_path = 'dataset/acoustic_data.txt'
    labels_file_path = 'dataset/geometry_data.txt'
    dataset = CustomDataset(data_file_path, labels_file_path)
    # Define the sizes for the training and test sets
    train_size = int(0.8 * len(dataset))  # 80% training
    test_size = len(dataset) - train_size  # 20% test
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # load training set and test set
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
  
    # Initialize the CNN
    input_size = 14
    output_size = 7
    cnn = CNN(input_size, output_size).to(device)
  
    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    # Run the training loop
    train_loss = []
    length_error = []
    width_error = []
    test_error = []
    test_error2 = []
    acc1 = []
    acc2 = []
    acc3 = []
    acc4 = []
    acc5 = []
    acc6 = []
    acc7 = []
    for epoch in range(0, 40): # 100 epochs at maximum
    
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
        
            # Get inputs
            inputs = data["data"].to(device)
            targets = data["label"].to(device)

            # Zero the gradients
            optimizer.zero_grad()
            
            # inputs = inputs.view(-1, 5, inputs.size(1))
            # Perform forward pass
            outputs = cnn(inputs)
            
            # Compute loss
            criterion = nn.MSELoss()
            loss = criterion(outputs, targets)
            
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            #if i % 500 == 499:
             #   print('Loss after mini-batch %5d: %.3f' %
             #           (i + 1, current_loss / 500))
             #   current_loss = 0.0
        current_loss /= len(train_loader)
        train_loss.append(current_loss)
        print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss))
        current_loss = 0

        # validation process
        accuracy = list()
        val_loss = 0
        for i, data in enumerate(test_loader, 0):
            # Get inputs
            inputs = data["data"].to(device)
            targets = data["label"].to(device)
            outputs = cnn(inputs)
            # calculate accuracy
            if i == 0:
                avg_accuracy = abs(outputs - targets) / targets * 100
            else:
                avg_accuracy += abs(outputs - targets) / targets * 100

            accuracy.append(abs(outputs - targets) / targets * 100)
            val_loss += criterion(outputs, targets).item()

        avg_accuracy /= len(test_loader)#test_size#len(test_loader)
        print(f'loss: {(val_loss/len(test_loader)):.4f}')
        print(f'Error:{avg_accuracy}')
        acc1.append(100-((avg_accuracy[4][0]).item()))
        acc2.append(100-((avg_accuracy[4][1]).item()))
        acc3.append(100-((avg_accuracy[4][2]).item()))
        acc4.append(100-((avg_accuracy[4][3]).item()))
        acc5.append(100-((avg_accuracy[4][4]).item()))
        acc6.append(100-((avg_accuracy[4][5]).item()))
        acc7.append(100-((avg_accuracy[4][6]).item()))
        length_error.append((avg_accuracy[4][0]).item())
        width_error.append((avg_accuracy[4][1]).item())
        test_error.append((val_loss/len(test_loader)))
        val_loss = 0
        

    # validation process
    accuracy = list()

    for i, data in enumerate(test_loader, 0):
        # Get inputs
        inputs = data["data"].to(device)
        targets = data["label"].to(device)
        outputs = cnn(inputs)
        # calculate accuracy
        if i == 0:
            avg_accuracy = abs(outputs - targets) / targets * 100
        else:
            avg_accuracy += abs(outputs - targets) / targets * 100

        accuracy.append(abs(outputs - targets) / targets * 100)
    avg_accuracy /= len(test_loader)
    print(avg_accuracy)

x = np.arange(1,41)
fig, ax =  plt.subplots(3, 2,figsize=(10,10)) # Creates figure fig and add an axes, ax.
fig.subplots_adjust(hspace=1.5)

ax[0,0].plot(x,np.asarray(train_loss),x,np.asarray(test_error))
ax[0,0].set_title("6 segtube Loss [CNN]")
ax[0,0].set_ylabel('MSE Loss')
ax[0,0].set_xlabel('# of Epoch')
ax[0,0].set_xlabel('# of Epoch')
ax[0,0].legend(['Train loss','Test loss'])

 
ax[1,0].plot(x,np.asarray(acc3))
ax[1,0].set_title("6 segtube Width Accuracy -2")
ax[1,0].set_ylabel('Percentage ACC')
ax[1,0].set_xlabel('# of Epoch')


ax[0,1].plot(x,np.asarray(acc2))
ax[0,1].set_title("6 segtube Width Accuracy -1")
ax[0,1].set_ylabel('Percentage ACC')
ax[0,1].set_xlabel('# of Epoch')


ax[1,1].plot(x,np.asarray(acc1))
ax[1,1].set_title("6 segtube Length Accuracy [CNN]")
ax[1,1].set_ylabel('Percentage Acc')
ax[1,1].set_xlabel('# of Epoch')

ax[2,0].plot(x,np.asarray(acc2),x,np.asarray(acc3),x,np.asarray(acc4),x,np.asarray(acc5),x,np.asarray(acc6),x,np.asarray(acc7))
ax[2,0].set_title("6 segtube Width Accuracy [CNN]")
ax[2,0].set_ylabel('Percentage ACC')
ax[2,0].set_xlabel('# of Epoch')
ax[2,0].legend(['Width-1','Width-2','Width-3','Width-4','Width-5','Width-6'])

plt.show()
    