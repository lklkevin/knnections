import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load(x_filename: str, y_filename: str):
    x = np.load(x_filename)
    y = np.load(y_filename)
    total = np.concatenate((y, x), axis=1)
    train, test = train_test_split(total, test_size=0.3, random_state=0)

    return train, test


class Autoencoder(torch.nn.Module):
    def __init__(self, start_dim: int, reduced_dim: int):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(start_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 16),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 32),
            # torch.nn.ReLU(),
            # torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, reduced_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(reduced_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(32, 64),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, start_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def optimize(start_dim, reduced_dim, epochs, inp, batch_size, lr=0.005, wd=0):
    model = Autoencoder(start_dim, reduced_dim)
    ls = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    losses = []
    data = torch.tensor(inp, dtype=torch.float32)

    batches = data.shape[0] // batch_size

    for ep in range(epochs):

        for b in range(batches):
            start = b * batch_size
            end = start + batch_size
            curr = data[start:end]

            reconstruction = model(curr)
            loss = ls(reconstruction, curr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            losses.append(loss)

    with torch.no_grad():
        latent = model.encoder(data).detach().numpy()

        return latent, losses


tr, tt = load('features.npy', 'labels.npy')
