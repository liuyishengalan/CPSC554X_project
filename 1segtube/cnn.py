import torch
from torch import nn
from generate_dataset import CustomDataset, CustomLoss
from torch.utils.data import DataLoader, random_split

# use cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128*2 , 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, x.size(0))
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = nn.functional.relu(x)
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc2(x)
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
    output_size = 2
    cnn = CNN(input_size, output_size).to(device)
  
    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    # Run the training loop
    for epoch in range(0, 50): # 100 epochs at maximum
    
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
            criterion = CustomLoss()
            loss = criterion(outputs, targets)
            
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0


        # validation process
        accuracy = list()

        for i, data in enumerate(test_loader, 0):
            # Get inputs
            inputs = data["data"].to(device)
            targets = data["label"].to(device)
            outputs = cnn(inputs)
            # print(targets.shape)
            # calculate accuracy
            if i == 0:
                avg_accuracy = abs(outputs - targets) / targets * 100
            else:
                avg_accuracy += abs(outputs - targets) / targets * 100

            accuracy.append(abs(outputs - targets) / targets * 100)
        avg_accuracy /= test_size
        print(avg_accuracy)

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
    avg_accuracy /= test_size
    print(avg_accuracy)
    