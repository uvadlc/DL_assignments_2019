import os
import errno

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision


class toTensor:
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 2:
            # Add channel dimension
            img = img[:, :, None]

        img = np.array(img).transpose(2, 0, 1)
        return torch.from_numpy(img)


class Flatten:
    def __init__(self):
        pass

    def __call__(self, img):
        return img.view(-1)


class toFloat:
    def __init__(self):
        pass

    def __call__(self, img):
        return img.float()


def mnist(root='./data/', batch_size=128, download=True):
    train_transforms = transforms.Compose([
        transforms.Pad(int(np.ceil(28 * 0.05)), padding_mode='edge'),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.CenterCrop(28),
        toTensor(),
        toFloat(),
        Flatten()
    ])
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        toTensor(),
        toFloat(),
        Flatten()
    ])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=train_transforms, target_transform=None,
        download=True)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, target_transform=None,
        download=True)

    train_dataset = data.dataset.Subset(dataset, np.arange(40000))
    val_dataset = data.dataset.Subset(dataset, np.arange(40000, 50000))

    trainloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader
