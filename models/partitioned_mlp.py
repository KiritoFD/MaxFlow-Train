import torch
import torch.nn as nn
from typing import List, Dict, Tuple

class PartitionedLinear(nn.Module):
    """Linear layer partitioned into sub-blocks for PFN analysis."""
    
    def __init__(self, in_features: int, out_features: int, num_blocks: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.block_size = out_features // num_blocks
        
        # Create sub-blocks as separate parameter groups
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, self.block_size, bias=(i == 0))
            for i in range(num_blocks)
        ])
        # Single bias for the entire layer
        if num_blocks > 1:
            self.bias = nn.Parameter(torch.zeros(out_features - self.block_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [block(x) for block in self.blocks]
        return torch.cat(outputs, dim=-1)
    
    def get_block_params(self, block_idx: int) -> List[nn.Parameter]:
        return list(self.blocks[block_idx].parameters())
    
    def get_block_gradients(self, block_idx: int) -> torch.Tensor:
        grads = []
        for p in self.blocks[block_idx].parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        if grads:
            return torch.cat(grads)
        # Return zero tensor on the same device as parameters
        device = next(self.blocks[block_idx].parameters()).device
        return torch.tensor([0.0], device=device)


class PartitionedMLP(nn.Module):
    """
    Three-layer MLP for MNIST with sub-block partitioning.
    - Input: 784
    - Hidden1: 512 (4 blocks of 128)
    - Hidden2: 256 (2 blocks of 128)
    - Output: 10
    """
    
    def __init__(self):
        super().__init__()
        self.layer1 = PartitionedLinear(784, 512, num_blocks=4)
        self.layer2 = PartitionedLinear(512, 256, num_blocks=2)
        self.layer3 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
        
        # Block configuration for PFN
        self.block_config = {
            'layer1': {'num_blocks': 4, 'layer': self.layer1},
            'layer2': {'num_blocks': 2, 'layer': self.layer2},
            'layer3': {'num_blocks': 1, 'layer': None}  # Output layer as single block
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        """Returns gradients for all blocks organized by layer."""
        gradients = {}
        device = next(self.parameters()).device
        
        # Layer 1 blocks
        gradients['layer1'] = [
            self.layer1.get_block_gradients(i) for i in range(4)
        ]
        
        # Layer 2 blocks
        gradients['layer2'] = [
            self.layer2.get_block_gradients(i) for i in range(2)
        ]
        
        # Layer 3 (single block)
        layer3_grads = []
        for p in self.layer3.parameters():
            if p.grad is not None:
                layer3_grads.append(p.grad.flatten())
        if layer3_grads:
            gradients['layer3'] = [torch.cat(layer3_grads)]
        else:
            gradients['layer3'] = [torch.tensor([0.0], device=device)]
        
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        """Returns parameter groups for optimizer with separate lr per block."""
        groups = []
        
        # Layer 1 blocks
        for i in range(4):
            groups.append({
                'params': self.layer1.get_block_params(i),
                'name': f'layer1_block{i}',
                'lr': 0.001
            })
        
        # Layer 2 blocks
        for i in range(2):
            groups.append({
                'params': self.layer2.get_block_params(i),
                'name': f'layer2_block{i}',
                'lr': 0.001
            })
        
        # Layer 3
        groups.append({
            'params': list(self.layer3.parameters()),
            'name': 'layer3_block0',
            'lr': 0.001
        })
        
        return groups
