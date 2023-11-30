import torch
from torch import nn
from generate_dataset import CustomDataset, CustomLoss
from torch.utils.data import DataLoader, random_split


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(14, 64),
      nn.ReLU(),
      nn.Linear(64, 16),
      nn.ReLU(),
      nn.Linear(16, 3)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
  
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  
    # Initialize the MLP
    mlp = MLP()
  
    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-5)

    # Run the training loop
    for epoch in range(0, 30): # 5 epochs at maximum
    
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
        
            # Get inputs
            inputs = data["data"]
            targets = data["label"]

            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
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

        for i, data in enumerate(test_dataset, 0):
            # Get inputs
            inputs = data["data"]
            targets = data["label"]
            outputs = mlp(inputs)
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

    for i, data in enumerate(test_dataset, 0):
        # Get inputs
        inputs = data["data"]
        targets = data["label"]
        outputs = mlp(inputs)
        # calculate accuracy
        if i == 0:
            avg_accuracy = abs(outputs - targets) / targets * 100
        else:
            avg_accuracy += abs(outputs - targets) / targets * 100

        accuracy.append(abs(outputs - targets) / targets * 100)
    avg_accuracy /= test_size
    print(avg_accuracy)
    