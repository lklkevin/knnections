import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load(x_filename: str, y_filename: str):
    x = np.load(x_filename)
    y = np.load(y_filename)
    total = np.concatenate((x, y), axis=1)
    train, test = train_test_split(total, test_size=0.3, random_state=0)

    return train, test


class Autoencoder(torch.nn.Module):
    def __init__(self, start_dim: int, reduced_dim: int):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(start_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, reduced_dim)
        )

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
        return self.decoder(self.encoder(x))


def optimize(start_dim, reduced_dim, epochs, inp, lr=0.005):
    model = Autoencoder(start_dim, reduced_dim)
    ls = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    data = torch.tensor(inp, dtype=torch.float32)

    for ep in range(epochs):
        curr = data

        reconstruction = model(curr)
        loss = ls(reconstruction, curr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    with torch.no_grad():
        latent = model.encoder(data).detach().numpy()

        return latent, losses


def grid_search(x_file, y_file):
    tr, _ = load(x_file, y_file)
    reduced_dims = [5, 10, 15]
    epochs_list = [10, 15, 20]
    learning_rates = [0.05, 0.01, 0.005]

    results = []
    for reduced_dim, epochs, lr in itertools.product(reduced_dims, epochs_list, learning_rates):
        total_loss = 0
        for i in range(tr.shape[0]):
            a, b = optimize(768, reduced_dim=reduced_dim, epochs=epochs, inp=tr[i], lr=lr)
            total_loss += b[-1].detach().item()
        average_loss = total_loss / tr.shape[0]
        results.append((reduced_dim, epochs, lr, average_loss))

    results.sort(key=lambda x: x[3])
    return results


if __name__ == "__main__":
    combos = grid_search('bert_ft_vect.npy', 'bert_lb_vect.npy')
    labels = [f'{t[0]}, {t[1]}, {t[2]}' for t in combos]
    heights = [t[3] for t in combos]

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, heights, tick_label=labels)
    plt.xlabel('Combination of Hyperparameters (Reduced Dim, Epochs, Learning Rate)')
    plt.ylabel('Average Loss')
    plt.title('Grid Search Results Sorted by Average Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Note we are only use the training set
    temp, _ = load('bert_ft_vect.npy', 'bert_lb_vect.npy')
    total_loss = np.zeros(15)

    for i in range(temp.shape[0]):
        reduced, loss = optimize(768, 5, 15, temp[i], 0.005)
        for j in range(15):
            total_loss[j] += loss[j].detach().item()

    avg_loss = total_loss / temp.shape[0]
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(15), avg_loss, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss vs Epoch for Autoencoder Training')
    plt.grid(True)
    plt.show()
