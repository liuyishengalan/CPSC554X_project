import torch
from torch import nn
from generate_dataset import CustomDataset, CustomLoss




class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(5, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, 2)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
  
if __name__ == '__main__':
    data_file_path = 'dataset/acoustic_data.txt'
    labels_file_path = 'dataset/geometry_data.txt'
    dataset = CustomDataset(data_file_path, labels_file_path)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    mlp = MLP()
  
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
    
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
        
            print(i)
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

    # Process is complete.
    print('Training process has finished.')