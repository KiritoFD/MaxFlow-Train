import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional

class NoisyDataset(Dataset):
    """Wrapper dataset that adds noise to images and/or labels."""
    
    def __init__(self, base_dataset: Dataset, 
                 pixel_noise_std: float = 0.0,
                 label_noise_ratio: float = 0.0,
                 num_classes: int = 10):
        """
        Args:
            base_dataset: Original dataset
            pixel_noise_std: Std of Gaussian noise added to pixels (0.0-1.0)
            label_noise_ratio: Fraction of labels to randomly flip (0.0-1.0)
            num_classes: Number of classes for label noise
        """
        self.base_dataset = base_dataset
        self.pixel_noise_std = pixel_noise_std
        self.label_noise_ratio = label_noise_ratio
        self.num_classes = num_classes
        
        # Pre-compute noisy labels
        self.noisy_labels = None
        if label_noise_ratio > 0:
            self._generate_noisy_labels()
    
    def _generate_noisy_labels(self):
        """Generate noisy labels by random flipping."""
        n = len(self.base_dataset)
        self.noisy_labels = []
        
        for i in range(n):
            _, label = self.base_dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if np.random.random() < self.label_noise_ratio:
                # Flip to a random different label
                new_label = np.random.randint(0, self.num_classes - 1)
                if new_label >= label:
                    new_label += 1
                self.noisy_labels.append(new_label)
            else:
                self.noisy_labels.append(label)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        
        # Add pixel noise
        if self.pixel_noise_std > 0:
            noise = torch.randn_like(image) * self.pixel_noise_std
            image = image + noise
            image = torch.clamp(image, -3, 3)  # Reasonable range for normalized data
        
        # Use noisy label if available
        if self.noisy_labels is not None:
            label = self.noisy_labels[idx]
        
        return image, label


class SmallSampleDataset(Dataset):
    """Dataset wrapper that limits samples per class."""
    
    def __init__(self, base_dataset: Dataset, samples_per_class: int = 100, 
                 num_classes: int = 10):
        """
        Args:
            base_dataset: Original dataset
            samples_per_class: Maximum samples to keep per class
            num_classes: Total number of classes
        """
        self.base_dataset = base_dataset
        
        # Collect indices per class
        class_indices = {i: [] for i in range(num_classes)}
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if len(class_indices[label]) < samples_per_class:
                class_indices[label].append(idx)
        
        # Flatten to single list
        self.indices = []
        for c in range(num_classes):
            self.indices.extend(class_indices[c])
        
        print(f"SmallSampleDataset: {len(self.indices)} samples "
              f"({samples_per_class} per class)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        return self.base_dataset[self.indices[idx]]
