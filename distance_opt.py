import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pca import apply_pca

num_epochs = 50
learning_rate = 0.001

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, PCA_DIMS):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(PCA_DIMS, PCA_DIMS * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(PCA_DIMS * 2, PCA_DIMS * 8)
        self.fc3 = nn.Linear(PCA_DIMS * 8, PCA_DIMS * 8)
        self.fc4 = nn.Linear(PCA_DIMS * 8, PCA_DIMS * 2)
        self.fc5 = nn.Linear(PCA_DIMS * 2, PCA_DIMS)

    def forward(self, x):
        # x = x.view(-1, x.shape[1])  # Flatten the input
        x = torch.from_numpy(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        return x

def train_and_get_model(X_train, y_train, PCA_DIMS = 7, ENABLE_PRINT = True):
    # Instantiate the model, define the loss function and the optimizer
    model = SimpleNN(PCA_DIMS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for day_inputs, day_labels in zip(X_train, y_train):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(day_inputs)
            loss = criterion(outputs, torch.from_numpy(day_labels))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            
        if ENABLE_PRINT:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(X_train) * len(X_train[0])}')
    
    return model
        
def evaluate_model(model, X_test, y_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in zip(X_test, y_test):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total}%')