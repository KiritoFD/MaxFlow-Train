"""
PFN Model Repository - 联合优化版
核心改进：恒定宽度分块 (Iso-Width Partitioning)
"""

import torch
import torch.nn as nn
from typing import List, Dict

def _safe_concat(grads: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Safely concatenate gradients or return zero tensor on empty list."""
    return torch.cat(grads) if grads else torch.tensor([0.0], device=device)

class PartitionedLinear(nn.Module):
    """Linear layer partitioned into sub-blocks for PFN analysis."""
    
    def __init__(self, in_features: int, out_features: int, num_blocks: int):
        super().__init__()
        assert out_features % num_blocks == 0, "Output features must be divisible by num_blocks"
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, out_features // num_blocks) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(x) for block in self.blocks], dim=-1)

class ChannelPartitionedConv2d(nn.Module):
    """
    联合优化版：支持 '自动分块' (Auto-Partitioning)
    保证每个 Block 的通道数接近 target_width，从而统一 PFN 的诊断颗粒度。
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, num_blocks: int = 4,
                 auto_partition: bool = True, target_width: int = 16):
        super().__init__()
        
        # === 联合优化核心逻辑 ===
        if auto_partition:
            # 动态计算分块数，保证每块至少有 target_width 个通道
            # 例如 512 通道 -> 32 块; 64 通道 -> 4 块
            num_blocks = max(1, out_channels // target_width)
            # 确保能整除
            while out_channels % num_blocks != 0 and num_blocks > 1:
                num_blocks -= 1
        
        assert out_channels % num_blocks == 0, f"out_channels {out_channels} must be divisible by num_blocks {num_blocks}"
        self.num_blocks = num_blocks
        self.block_size = out_channels // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels, self.block_size, kernel_size, stride, padding)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(x) for block in self.blocks], dim=1)

class PartitionedCNN(nn.Module):
    """专为PFN设计的分块CNN - 支持自动分块"""
    
    def __init__(self, num_classes: int = 10, num_blocks: int = 4, use_bn: bool = True,
                 auto_partition: bool = True, target_width: int = 16):
        super().__init__()
        self.use_bn = use_bn
        self.auto_partition = auto_partition
        
        self.layer1 = ChannelPartitionedConv2d(3, 64, 3, padding=1, num_blocks=num_blocks,
                                                auto_partition=auto_partition, target_width=target_width)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else None
        
        self.layer2 = ChannelPartitionedConv2d(64, 128, 3, padding=1, num_blocks=num_blocks,
                                                auto_partition=auto_partition, target_width=target_width)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else None
        
        self.layer3 = ChannelPartitionedConv2d(128, 256, 3, padding=1, num_blocks=num_blocks,
                                                auto_partition=auto_partition, target_width=target_width)
        self.bn3 = nn.BatchNorm2d(256) if use_bn else None
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        
        # 记录实际分块数（用于 get_parameter_groups）
        self._layers = [self.layer1, self.layer2, self.layer3]

    def forward(self, x):
        x = self.layer1(x)
        if self.bn1: x = self.bn1(x)
        x = self.pool(self.relu(x))
        
        x = self.layer2(x)
        if self.bn2: x = self.bn2(x)
        x = self.pool(self.relu(x))
        
        x = self.layer3(x)
        if self.bn3: x = self.bn3(x)
        x = self.pool(self.relu(x))
        
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients: Dict[str, List[torch.Tensor]] = {}
        device = next(self.parameters()).device
        
        for idx, layer in enumerate(self._layers, start=1):
            block_grads = []
            for block in layer.blocks:
                grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
                block_grads.append(_safe_concat(grads, device))
            gradients[f'conv{idx}'] = block_grads
        
        fc_grads = [p.grad.flatten() for p in self.fc.parameters() if p.grad is not None]
        gradients['fc'] = [_safe_concat(fc_grads, device)]
        return gradients

    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        for idx, layer in enumerate(self._layers, start=1):
            for block_idx in range(layer.num_blocks):
                groups.append({
                    'params': list(layer.blocks[block_idx].parameters()),
                    'name': f'conv{idx}_block{block_idx}',
                    'lr': 0.001
                })
        
        if self.use_bn:
            bn_params = []
            for bn in [self.bn1, self.bn2, self.bn3]:
                if bn: bn_params.extend(list(bn.parameters()))
            if bn_params:
                groups.append({'params': bn_params, 'name': 'bn_all', 'lr': 0.001})
        
        groups.append({'params': list(self.fc.parameters()), 'name': 'fc_block0', 'lr': 0.001})
        return groups

class DeepPartitionedCNN(nn.Module):
    """
    联合优化版深层CNN：恒定宽度分块 + 无残差/无BN
    真正的PFN压力测试场景
    """
    
    def __init__(self, num_layers: int = 12, num_classes: int = 10, num_blocks: int = 4,
                 auto_partition: bool = True, target_width: int = 16):
        super().__init__()
        self.auto_partition = auto_partition
        self.target_width = target_width
        
        self.conv_layers = nn.ModuleList()
        in_ch = 3
        curr_ch = 64
        
        for i in range(num_layers):
            layer = ChannelPartitionedConv2d(
                in_ch, curr_ch, 3, padding=1, num_blocks=num_blocks,
                auto_partition=auto_partition, target_width=target_width
            )
            self.conv_layers.append(layer)
            in_ch = curr_ch
            if (i + 1) % 4 == 0:
                curr_ch = min(curr_ch * 2, 512)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        for idx, layer in enumerate(self.conv_layers):
            x = self.relu(layer(x))
            if (idx + 1) % 4 == 0 and x.shape[-1] > 2:
                x = self.pool(x)
        return self.fc(self.gap(x).view(x.size(0), -1))

    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        gradients: Dict[str, List[torch.Tensor]] = {}
        device = next(self.parameters()).device
        
        for idx, layer in enumerate(self.conv_layers):
            block_grads = []
            for block in layer.blocks:
                grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
                block_grads.append(_safe_concat(grads, device))
            gradients[f'layer{idx}'] = block_grads
        
        fc_grads = [p.grad.flatten() for p in self.fc.parameters() if p.grad is not None]
        gradients['fc'] = [_safe_concat(fc_grads, device)]
        return gradients

    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        for layer_idx, layer in enumerate(self.conv_layers):
            for block_idx in range(layer.num_blocks):
                groups.append({
                    'params': list(layer.blocks[block_idx].parameters()),
                    'name': f'L{layer_idx}_B{block_idx}',
                    'lr': 0.001
                })
        groups.append({'params': list(self.fc.parameters()), 'name': 'fc', 'lr': 0.001})
        return groups

def get_model(scenario: str, dataset: str, **kwargs) -> nn.Module:
    """
    Model factory for PFN experiments
    新增参数：auto_partition, target_width
    """
    num_classes = 100 if dataset == 'cifar100' else 10
    if dataset not in {'cifar10', 'cifar100'}:
        raise ValueError(f"Only CIFAR datasets supported, got {dataset}")
    
    auto_partition = kwargs.get('auto_partition', True)
    target_width = kwargs.get('target_width', 16)
    
    if scenario == 'standard':
        return PartitionedCNN(
            num_classes=num_classes,
            num_blocks=kwargs.get('num_blocks', 4),
            use_bn=True,
            auto_partition=auto_partition,
            target_width=target_width
        )
    elif scenario == 'standard_no_bn':
        return PartitionedCNN(
            num_classes=num_classes,
            num_blocks=kwargs.get('num_blocks', 4),
            use_bn=False,
            auto_partition=auto_partition,
            target_width=target_width
        )
    elif scenario == 'deep':
        return DeepPartitionedCNN(
            num_layers=kwargs.get('num_layers', 12),
            num_classes=num_classes,
            num_blocks=kwargs.get('num_blocks', 4),
            auto_partition=auto_partition,
            target_width=target_width
        )
    elif scenario == 'bottleneck':
        return DeepPartitionedCNN(
            num_layers=kwargs.get('num_layers', 8),
            num_classes=num_classes,
            num_blocks=kwargs.get('num_blocks', 4),
            auto_partition=auto_partition,
            target_width=target_width
        )
    raise ValueError(f"Unknown scenario: {scenario}")