import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from vectorize import load_vect as load
from pca import apply_pca
import numpy as np
import matplotlib.pyplot as plt


# Define a simple feedforward neural network
class DistOptimizer(nn.Module):
    def __init__(self, dims):
        super(DistOptimizer, self).__init__()
        self.fc1 = nn.Linear(dims, dims * 2)
        self.fc2 = nn.Linear(dims * 2, dims * 8)
        self.fc3 = nn.Linear(dims * 8, dims * 8)
        self.fc4 = nn.Linear(dims * 8, dims * 2)
        self.fc5 = nn.Linear(dims * 2, dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        
        return x


# Default params are optimized via grid search
def train_and_get_model(X_train, y_train, dims = 8, ep = 50, lr = 0.005, get_loss = False):
    # Instantiate the model, define the loss function and the optimizer
    num_epochs = ep
    learning_rate = lr
    model = DistOptimizer(dims)
    lossf = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for _ in range(num_epochs):
        for day_inputs, day_labels in zip(X_train, y_train):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(day_inputs)
            loss = lossf(outputs, torch.from_numpy(day_labels))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            losses.append(loss.detach().item())
    
    if get_loss:
        return losses

    return model


def dist_opt_gridsearch(x_file, y_file):
    tr, _ = load(x_file, y_file, merge=True)
    epochs_list = [25, 50, 75]
    learning_rates = [0.01, 0.005, 0.001]
    results = []

    # Go through all combinations of hyperparameters
    for epochs, lr in itertools.product(epochs_list, learning_rates):
        total_loss = 0
        for i in range(tr.shape[0]):
            curr = tr[i]
            curr, _ = apply_pca(curr, 8)
            X_train, y_train = curr[:16], curr[16:]
            b = train_and_get_model(X_train, y_train, 8, epochs, lr, True)
            total_loss += b[-1]
        average_loss = total_loss / tr.shape[0]
        print(average_loss)
        results.append((epochs, lr, average_loss))

    # Sort by the average loss

    # Results from before saved below:
    # results = [(25, 0.01, 2.186841199035142), (25, 0.005, 2.154186282154857), (25, 0.001, 2.8758281364428755), (50, 0.01, 2.1694160461501806), (50, 0.005, 2.0839534693391246), (50, 0.001, 2.283987769109457), (75, 0.01, 2.1673604639462467), (75, 0.005, 2.0523286657217814), (75, 0.001, 2.125368909279936)]
    results.sort(key=lambda x: x[2])
    labels = [f'{t[0]}, {t[1]}' for t in results]
    heights = [t[2] for t in results]

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, heights, tick_label=labels)
    plt.xlabel('Combination of Hyperparameters (Epochs, Learning Rate)')
    plt.ylabel('Average Loss')
    plt.title('Distance Optimization NN Grid Search Sorted by MSE in BERT Vectorized Training Data')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dist_opt_gridsearch("./data/bert_ft_vect.npy", "./data/bert_lb_vect.npy")