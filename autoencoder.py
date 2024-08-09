import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from vectorize import load_vect as load


class Autoencoder(torch.nn.Module):
    def __init__(self, start_dim: int, reduced_dim: int):
        super().__init__()

        # Encode to a lower dimension
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(start_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, reduced_dim)
        )

        # Decode back to the original dimension
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(reduced_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, start_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Each forward pass is basically decoding the encoded input
        return self.decoder(self.encoder(x))


def optimize(start_dim, inp, reduced_dim=5, epochs=11, lr=0.005):
    # Defining the model, loss function, optimzer
    model = Autoencoder(start_dim, reduced_dim)
    ls = torch.nn.MSELoss()
    # Adam is a GD method
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    data = torch.tensor(inp, dtype=torch.float32)

    for _ in range(epochs):
        # Full-batch used
        curr = data

        # Forward pass, calculate loss, backwards pass
        reconstruction = model(curr)
        loss = ls(reconstruction, curr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)

    # Return latent representation of data
    with torch.no_grad():
        latent = model.encoder(data).detach().numpy()

        return latent, losses


def grid_search(x_file, y_file):
    tr, _ = load(x_file, y_file, merge=True)
    reduced_dims = [5, 10, 15]
    epochs_list = [10, 15, 20]
    learning_rates = [0.05, 0.01, 0.005]
    results = []

    # Go through all combinations of hyperparameters
    for reduced_dim, epochs, lr in itertools.product(reduced_dims, epochs_list, learning_rates):
        total_loss = 0
        for i in range(tr.shape[0]):
            _, b = optimize(768, inp=tr[i], reduced_dim=reduced_dim, epochs=epochs, lr=lr)
            total_loss += b[-1].detach().item()
        average_loss = total_loss / tr.shape[0]
        results.append((reduced_dim, epochs, lr, average_loss))

    # Sort by the average loss
    results.sort(key=lambda x: x[3])
    return results


if __name__ == "__main__":
    combos = grid_search('./data/bert_ft_vect.npy', './data/bert_lb_vect.npy')
    labels = [f'{t[0]}, {t[1]}, {t[2]}' for t in combos]
    heights = [t[3] for t in combos]

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, heights, tick_label=labels)
    plt.xlabel('Combination of Hyperparameters (Reduced Dim, Epochs, Learning Rate)')
    plt.ylabel('Average Loss')
    plt.title('Grid Search Results Sorted by Average Loss in BERT Vectorized Training Data')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
