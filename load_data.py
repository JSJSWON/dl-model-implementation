import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def load_data(batch_size, num_workers, pin_memory, image_size, val_ratio, shuffle, seed):

    # transform
    train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    

    # dataset
    train = datasets.CIFAR10(root='CIFAR10/', train=True, transform=train_transform, download=True)
    val = datasets.CIFAR10(root='CIFAR10/', train=True, transform=test_transform, download=True)
    test = datasets.CIFAR10(root='CIFAR10/', train=False, transform=test_transform, download=True)


    # split train/val
    # 참고: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    num_train = len(train)
    indices = list(range(num_train))
    split = int(val_ratio * num_train)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)


    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
