from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def build_transforms(use_autoaugment=False):
    normalize = transforms.Normalize(MEAN, STD)
    train_tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)  # set use_autoaugment=True to switch to AutoAugment(CIFAR10)
    ]
    if use_autoaugment:
        train_tfms[-1] = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
    train_tfms += [transforms.ToTensor(), normalize]
    test_tfms = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_tfms), transforms.Compose(test_tfms)

def get_dataloaders(data_root="./data", batch_size=128, num_workers=2, val_ratio=0.1, seed=42, use_autoaugment=False):
    train_tfms, test_tfms = build_transforms(use_autoaugment=use_autoaugment)
    train_full = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_tfms)
    test_set   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)

    val_size = int(len(train_full) * val_ratio)
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [len(train_full)-val_size, val_size], generator=g)

    mk = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  **mk),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False, **mk),
        DataLoader(test_set,  batch_size=batch_size, shuffle=False, **mk),
    )
