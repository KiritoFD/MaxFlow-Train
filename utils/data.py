import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
from .noise import NoisyDataset, SmallSampleDataset


def get_dataloaders(
    dataset: str = 'mnist',
    batch_size: int = 64,
    pixel_noise: float = 0.0,
    label_noise: float = 0.0,
    samples_per_class: Optional[int] = None,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Get train and test dataloaders for specified dataset.
    
    Args:
        dataset: 'mnist' or 'cifar10'
        batch_size: Batch size
        pixel_noise: Gaussian noise std
        label_noise: Label flip ratio
        samples_per_class: Limit samples per class
        data_dir: Data directory
        
    Returns:
        train_loader, test_loader, num_classes
    """
    
    if dataset.lower() == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, 
                                       transform=transform_train)
        test_dataset = datasets.MNIST(data_dir, train=False, 
                                      transform=transform_test)
        num_classes = 10
        
    elif dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transform_test)
        num_classes = 10
        
    elif dataset.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Apply data modifications
    if samples_per_class:
        train_dataset = SmallSampleDataset(train_dataset, samples_per_class, num_classes)
    
    if pixel_noise > 0 or label_noise > 0:
        train_dataset = NoisyDataset(train_dataset, pixel_noise, label_noise, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, num_classes
