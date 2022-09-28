import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

class SequentialMNIST(Dataset):

    def __init__(self, MNIST_dataset: torchvision.datasets.MNIST):
        self.MNIST_dataset = MNIST_dataset

    def __len__(self):
        return len(self.MNIST_dataset)

    def __getitem__(self, idx: int):
        image, target = self.MNIST_dataset[idx]

        # Return flattened image in shape (time steps = 784, dimensionality = 1)
        return image.view(-1, 1), target
