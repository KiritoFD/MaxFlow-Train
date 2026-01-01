"""
Unified Model Collection for PFN Experiments
Includes: PartitionedMLP, BottleneckMLP, DeepMLP, StandardCNN, BottleneckCNN, DeepCNN, PartitionedCNN
"""

import torch
import torch.nn as nn
from typing import List, Dict


# ==================== MLP MODELS ====================

class PartitionedLinear(nn.Module):
    """Linear layer partitioned into sub-blocks for PFN analysis."""
    
    def __init__(self, in_features: int, out_features: int, num_blocks: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.block_size = out_features // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, self.block_size, bias=True)
            for i in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(x) for block in self.blocks], dim=-1)
    
    def get_block_params(self, block_idx: int) -> List[nn.Parameter]:
        return list(self.blocks[block_idx].parameters())
    
    def get_block_gradients(self, block_idx: int) -> torch.Tensor:
        grads = [p.grad.flatten() for p in self.blocks[block_idx].parameters() if p.grad is not None]
        if grads: return torch.cat(grads)
        return torch.tensor([0.0], device=next(self.blocks[block_idx].parameters()).device)


class PartitionedMLP(nn.Module):
    """
    Enhanced MLP for MNIST with sub-block partitioning and Dropout.
    - Input: 784
    - Hidden1: 512 (4 blocks of 128) -> ReLU -> Dropout
    - Hidden2: 256 (2 blocks of 128) -> ReLU -> Dropout
    - Output: 10
    """
    
    def __init__(self):
        super().__init__()
        self.layer1 = PartitionedLinear(784, 512, num_blocks=4)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = PartitionedLinear(512, 256, num_blocks=2)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = self.dropout1(self.relu(self.layer1(x)))
        x = self.dropout2(self.relu(self.layer2(x)))
        return self.layer3(x)
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients = {}
        device = next(self.parameters()).device
        gradients['layer1'] = [self.layer1.get_block_gradients(i) for i in range(4)]
        gradients['layer2'] = [self.layer2.get_block_gradients(i) for i in range(2)]
        grads = [p.grad.flatten() for p in self.layer3.parameters() if p.grad is not None]
        gradients['layer3'] = [torch.cat(grads)] if grads else [torch.tensor([0.0], device=device)]
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        for i in range(4): groups.append({'params': self.layer1.get_block_params(i), 'name': f'layer1_block{i}', 'lr': 0.001})
        for i in range(2): groups.append({'params': self.layer2.get_block_params(i), 'name': f'layer2_block{i}', 'lr': 0.001})
        groups.append({'params': list(self.layer3.parameters()), 'name': 'layer3_block0', 'lr': 0.001})
        return groups


class BottleneckMLP(nn.Module):
    """
    Hourglass MLP with artificial bottleneck layers.
    Architecture: 784 -> 256 -> 64 -> 8 -> 64 -> 256 -> 10
    The narrow 8-neuron layer (reduced from 16) creates a strong information bottleneck.
    """
    
    def __init__(self, bottleneck_width: int = 8):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = nn.Linear(784, 256), nn.Linear(256, 64), nn.Linear(64, bottleneck_width)
        self.dec1, self.dec2, self.output = nn.Linear(bottleneck_width, 64), nn.Linear(64, 256), nn.Linear(256, 10)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0.1)
        self.layers = [self.enc1, self.enc2, self.enc3, self.dec1, self.dec2, self.output]
        self.layer_names = ['enc1', 'enc2', 'bottleneck', 'dec1', 'dec2', 'output']
    
    def forward(self, x):
        x = x.view(-1, 784)
        for i, layer in enumerate(self.layers[:-1]):
            x = self.dropout(self.relu(layer(x))) if i != 2 else self.relu(layer(x))
        return self.output(x)
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients = {}
        device = next(self.parameters()).device
        for name, layer in zip(self.layer_names, self.layers):
            grads = [p.grad.flatten() for p in layer.parameters() if p.grad is not None]
            gradients[name] = [torch.cat(grads)] if grads else [torch.tensor([0.0], device=device)]
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        return [{'params': list(layer.parameters()), 'name': f'{name}_block0', 'lr': 0.001}
                for name, layer in zip(self.layer_names, self.layers)]


class DeepMLP(nn.Module):
    """
    Very deep MLP (10+ layers) without residual connections.
    This will suffer from gradient vanishing, perfect for PFN to help.
    """
    
    def __init__(self, num_hidden_layers: int = 10, hidden_dim: int = 128):
        super().__init__()
        layers = [nn.Linear(784, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)] + [nn.Linear(hidden_dim, 10)]
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.layer_names = [f'hidden_{i}' for i in range(num_hidden_layers)] + ['output']
    
    def forward(self, x):
        x = x.view(-1, 784)
        for layer in self.layers[:-1]: x = self.relu(layer(x))
        return self.layers[-1](x)
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients = {}
        device = next(self.parameters()).device
        for name, layer in zip(self.layer_names, self.layers):
            grads = [p.grad.flatten() for p in layer.parameters() if p.grad is not None]
            gradients[name] = [torch.cat(grads)] if grads else [torch.tensor([0.0], device=device)]
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        return [{'params': list(layer.parameters()), 'name': f'{name}_block0', 'lr': 0.001}
                for name, layer in zip(self.layer_names, self.layers)]


# ==================== CNN MODELS ====================

class ChannelPartitionedConv2d(nn.Module):
    """将卷积层沿输出通道拆分为多个独立Block，构建并行流路径"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, num_blocks: int = 4):
        super().__init__()
        assert out_channels % num_blocks == 0
        self.num_blocks = num_blocks
        self.block_size = out_channels // num_blocks
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels, self.block_size, kernel_size, stride, padding)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(x) for block in self.blocks], dim=1)


class PartitionedCNN(nn.Module):
    """专为PFN设计的分块CNN，每层4个并行流管道"""
    
    def __init__(self, num_blocks: int = 4, num_classes: int = 10):
        super().__init__()
        self.num_blocks = num_blocks
        self.conv1 = ChannelPartitionedConv2d(3, 64, 3, padding=1, num_blocks=num_blocks)
        self.conv2 = ChannelPartitionedConv2d(64, 128, 3, padding=1, num_blocks=num_blocks)
        self.conv3 = ChannelPartitionedConv2d(128, 128, 3, padding=1, num_blocks=num_blocks)
        self.relu, self.pool, self.gap = nn.ReLU(), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.layer_names = ['conv1', 'conv2', 'conv3']
        self._layers = [self.conv1, self.conv2, self.conv3]
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(self.gap(x).view(x.size(0), -1))
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients = {}
        device = next(self.parameters()).device
        for name, layer in zip(self.layer_names, self._layers):
            block_grads = []
            for block in layer.blocks:
                grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
                block_grads.append(torch.cat(grads) if grads else torch.tensor([0.0], device=device))
            gradients[name] = block_grads
        fc_grads = [p.grad.flatten() for p in self.fc.parameters() if p.grad is not None]
        gradients['fc'] = [torch.cat(fc_grads)] if fc_grads else [torch.tensor([0.0], device=device)]
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        for name, layer in zip(self.layer_names, self._layers):
            for i, block in enumerate(layer.blocks):
                groups.append({'params': list(block.parameters()), 'name': f'{name}_block{i}', 'lr': 0.001})
        groups.append({'params': list(self.fc.parameters()), 'name': 'fc_block0', 'lr': 0.001})
        return groups


class DeepPartitionedCNN(nn.Module):
    """
    真正的PFN主场：既深（梯度消失），又宽（分块路由）。
    无残差连接，无BatchNorm，制造梯度消失瓶颈让PFN发挥作用。
    """
    
    def __init__(self, num_layers: int = 12, num_classes: int = 10, num_blocks: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        
        # 初始层
        self.conv_in = ChannelPartitionedConv2d(3, 32, 3, padding=1, num_blocks=num_blocks)
        
        # 中间深层
        self.hidden_layers = nn.ModuleList()
        
        in_ch, out_ch = 32, 32
        for i in range(num_layers - 1):
            if i > 0 and i % 4 == 0:
                out_ch = min(in_ch * 2, 128)
            self.hidden_layers.append(ChannelPartitionedConv2d(in_ch, out_ch, 3, padding=1, num_blocks=num_blocks))
            in_ch = out_ch
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_ch, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
        self.layer_names = ['conv_in'] + [f'conv_{i}' for i in range(len(self.hidden_layers))] + ['classifier']
    
    def forward(self, x):
        x = self.relu(self.conv_in(x))
        
        for i, conv in enumerate(self.hidden_layers):
            x = self.relu(conv(x))
            if i > 0 and i % 4 == 0:
                x = self.maxpool(x)
        
        x = self.pool(x)
        return self.classifier(x.view(x.size(0), -1))
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients = {}
        device = next(self.parameters()).device
        
        # conv_in
        block_grads = []
        for block in self.conv_in.blocks:
            grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
            block_grads.append(torch.cat(grads) if grads else torch.tensor([0.0], device=device))
        gradients['conv_in'] = block_grads
        
        # 中间层
        for i, layer in enumerate(self.hidden_layers):
            block_grads = []
            for block in layer.blocks:
                grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
                block_grads.append(torch.cat(grads) if grads else torch.tensor([0.0], device=device))
            gradients[f'conv_{i}'] = block_grads
        
        # Classifier
        fc_grads = [p.grad.flatten() for p in self.classifier.parameters() if p.grad is not None]
        gradients['classifier'] = [torch.cat(fc_grads)] if fc_grads else [torch.tensor([0.0], device=device)]
        
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        
        # conv_in
        for i, block in enumerate(self.conv_in.blocks):
            groups.append({'params': list(block.parameters()), 'name': f'conv_in_block{i}', 'lr': 0.001})
        
        # 中间层
        for layer_idx, layer in enumerate(self.hidden_layers):
            for i, block in enumerate(layer.blocks):
                groups.append({'params': list(block.parameters()), 'name': f'conv_{layer_idx}_block{i}', 'lr': 0.001})
        
        groups.append({'params': list(self.classifier.parameters()), 'name': 'classifier_block0', 'lr': 0.001})
        
        return groups


class BottleneckCNN(nn.Module):
    """
    CNN with strong bottleneck architecture for CIFAR-10.
    Creates artificial information bottleneck to test PFN.
    No BatchNorm, reduced bottleneck width (4 instead of 8).
    """
    
    def __init__(self, bottleneck_channels: int = 4, num_classes: int = 10):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.bottleneck = nn.Sequential(nn.Conv2d(128, bottleneck_channels, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(bottleneck_channels, 128, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(256, num_classes)
        self.layer_names = ['enc1', 'enc2', 'bottleneck', 'dec1', 'dec2', 'classifier']
        self._layers = [self.enc1, self.enc2, self.bottleneck, self.dec1, self.dec2, self.classifier]
    
    def forward(self, x):
        x = self.dec2(self.dec1(self.bottleneck(self.enc2(self.enc1(x)))))
        return self.classifier(x.view(x.size(0), -1))
    
    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients = {}
        device = next(self.parameters()).device
        for name, layer in zip(self.layer_names, self._layers):
            grads = [p.grad.flatten() for p in layer.parameters() if p.grad is not None]
            gradients[name] = [torch.cat(grads)] if grads else [torch.tensor([0.0], device=device)]
        return gradients
    
    def get_parameter_groups(self) -> List[Dict]:
        return [{'params': list(layer.parameters()), 'name': f'{name}_block0', 'lr': 0.001}
                for name, layer in zip(self.layer_names, self._layers)]


# ==================== MODEL FACTORY ====================

def get_model(scenario: str, dataset: str, **kwargs) -> nn.Module:
    num_classes = 100 if dataset == 'cifar100' else 10
    
    if dataset == 'mnist':
        if scenario == 'standard': return PartitionedMLP()
        elif scenario == 'bottleneck': return BottleneckMLP(bottleneck_width=kwargs.get('bottleneck_width', 8))
        elif scenario == 'deep': return DeepMLP(num_hidden_layers=kwargs.get('num_layers', 10))
    
    elif dataset in ['cifar10', 'cifar100']:
        if scenario == 'standard': 
            return PartitionedCNN(num_blocks=4, num_classes=num_classes)
        elif scenario == 'bottleneck': 
            return BottleneckCNN(bottleneck_channels=kwargs.get('bottleneck_width', 4), num_classes=num_classes)
        elif scenario == 'deep': 
            return DeepPartitionedCNN(num_layers=kwargs.get('num_layers', 12), num_classes=num_classes, num_blocks=4)
    
    raise ValueError(f"Unknown scenario '{scenario}' or dataset '{dataset}'")