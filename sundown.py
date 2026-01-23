from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_set = datasets.Places365(root="./data", download=True)
val_set = datasets.Places365(root="./data", download=True)