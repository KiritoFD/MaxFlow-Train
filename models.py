#%%writefile models.py
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
        self.out_features = out_features  # expose for external access
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, out_features // num_blocks) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(x) for block in self.blocks], dim=-1)

class ChannelPartitionedConv2d(nn.Module):
    """
    联合优化版：支持自动分块 (Auto-Partitioning)
    保证每个Block的通道数接近target_width，统一PFN的诊断颗粒度
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, num_blocks: int = 4,
                 auto_partition: bool = True, target_width: int = 16):
        super().__init__()
        
        # 联合优化核心：动态计算分块数
        if auto_partition:
            # 动态计算分块数，保证每块至少有target_width个通道
            num_blocks = max(1, out_channels // target_width)
            # 确保能整除
            while out_channels % num_blocks != 0 and num_blocks > 1:
                num_blocks -= 1
        else:
            # 传统模式：确保能整除
            assert out_channels % num_blocks == 0, "Output channels must be divisible by num_blocks"
        
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
        
        # 使用自动分块时，各层分块数会不同
        self.layer1 = ChannelPartitionedConv2d(3, 64, 3, padding=1, 
                                                num_blocks=num_blocks, 
                                                auto_partition=auto_partition, 
                                                target_width=target_width)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else None
        
        self.layer2 = ChannelPartitionedConv2d(64, 128, 3, padding=1,
                                                num_blocks=num_blocks,
                                                auto_partition=auto_partition,
                                                target_width=target_width)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else None
        
        self.layer3 = ChannelPartitionedConv2d(128, 256, 3, padding=1,
                                                num_blocks=num_blocks,
                                                auto_partition=auto_partition,
                                                target_width=target_width)
        self.bn3 = nn.BatchNorm2d(256) if use_bn else None
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        
        # 记录实际分块数（用于调试）
        self._log_partition_info()
    
    def _log_partition_info(self):
        """打印分块信息"""
        print(f"  [PartitionedCNN] Auto-partition={self.auto_partition}")
        print(f"    Layer1: {self.layer1.num_blocks} blocks × {self.layer1.block_size} ch")
        print(f"    Layer2: {self.layer2.num_blocks} blocks × {self.layer2.block_size} ch")
        print(f"    Layer3: {self.layer3.num_blocks} blocks × {self.layer3.block_size} ch")

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
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3], start=1):
            block_grads = []
            for block in layer.blocks:
                grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
                block_grads.append(_safe_concat(grads, device))
            gradients[f'conv{idx}'] = block_grads
        fc_grads = [p.grad.flatten() for p in self.fc.parameters() if p.grad is not None]
        gradients['fc'] = [_safe_concat(fc_grads, device)]
        return gradients

    def get_layer_widths(self) -> Dict[str, int]:
        """返回各层的输出宽度（用于Sink容量校正）"""
        return {
            'conv1': 64,
            'conv2': 128,
            'conv3': 256,
            'fc': self.fc.out_features  # FC层宽度通常较小
        }

    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3], start=1):
            for block_idx in range(layer.num_blocks):  # 使用实际分块数
                groups.append({
                    'params': list(layer.blocks[block_idx].parameters()),
                    'name': f'conv{idx}_block{block_idx}',
                    'lr': 0.001
                })
        if self.use_bn:
            bn_params = list(self.bn1.parameters()) + list(self.bn2.parameters()) + list(self.bn3.parameters())
            groups.append({'params': bn_params, 'name': 'bn_all', 'lr': 0.001})
        groups.append({'params': list(self.fc.parameters()), 'name': 'fc_block0', 'lr': 0.001})
        return groups


class DeepPartitionedCNN(nn.Module):
    """
    深层PFN模型 - 支持自动分块
    无残差连接，制造梯度消失瓶颈让PFN发挥作用
    """
    
    def __init__(self, num_layers: int = 12, num_classes: int = 10, num_blocks: int = 4,
                 auto_partition: bool = True, target_width: int = 16):
        super().__init__()
        self.auto_partition = auto_partition
        self.conv_layers = nn.ModuleList()
        
        in_ch = 3
        curr_ch = 64
        for i in range(num_layers):
            layer = ChannelPartitionedConv2d(
                in_ch, curr_ch, 3, padding=1,
                num_blocks=num_blocks,
                auto_partition=auto_partition,
                target_width=target_width
            )
            self.conv_layers.append(layer)
            in_ch = curr_ch
            if (i + 1) % 4 == 0:
                curr_ch = min(curr_ch * 2, 512)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        
        self._log_partition_info()
    
    def _log_partition_info(self):
        """打印分块信息"""
        print(f"  [DeepPartitionedCNN] {len(self.conv_layers)} layers, Auto-partition={self.auto_partition}")
        for i, layer in enumerate(self.conv_layers):
            if i < 3 or i >= len(self.conv_layers) - 2:  # 只打印前3层和后2层
                print(f"    Layer{i}: {layer.num_blocks} blocks × {layer.block_size} ch")
            elif i == 3:
                print(f"    ... (middle layers) ...")

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

    def get_layer_widths(self) -> Dict[str, int]:
        """返回各层的输出宽度（用于Sink容量校正）"""
        widths = {}
        for idx, layer in enumerate(self.conv_layers):
            widths[f'layer{idx}'] = layer.block_size * layer.num_blocks
        widths['fc'] = self.fc.out_features
        return widths

    def get_parameter_groups(self) -> List[Dict]:
        groups = []
        for layer_idx, layer in enumerate(self.conv_layers):
            for block_idx in range(layer.num_blocks):  # 使用实际分块数
                groups.append({
                    'params': list(layer.blocks[block_idx].parameters()),
                    'name': f'L{layer_idx}_B{block_idx}',
                    'lr': 0.001
                })
        groups.append({'params': list(self.fc.parameters()), 'name': 'fc', 'lr': 0.001})
        return groups


class HourglassBottleneckCNN(nn.Module):
    """
    模拟沙漏 (Hourglass) 瓶颈结构
    Encoder -> Narrow Bottleneck -> Decoder -> Classifier
    设计用于 CIFAR (32x32) 上让瓶颈在空间和通道维度上都非常窄。
    """
    def __init__(self, num_classes: int = 10, bottleneck_width: int = 2, target_width: int = 16):
        super().__init__()
        # Encoder: 32x32 -> 16x16 -> 8x8
        self.enc1 = ChannelPartitionedConv2d(3, 32, kernel_size=3, stride=2, padding=1, target_width=target_width)
        self.enc2 = ChannelPartitionedConv2d(32, 64, kernel_size=3, stride=2, padding=1, target_width=target_width)

        # Bottleneck: 8x8x64 -> 8x8xbottleneck_width
        self.bottleneck = ChannelPartitionedConv2d(64, bottleneck_width, kernel_size=1, stride=1, padding=0,
                                                   target_width=max(1, bottleneck_width))

        # Decoder: upsample back to 32x32
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ChannelPartitionedConv2d(bottleneck_width, 32, kernel_size=3, stride=1, padding=1, target_width=target_width)
        self.dec2 = ChannelPartitionedConv2d(32, 64, kernel_size=3, stride=1, padding=1, target_width=target_width)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # classifier head: partitioned linear for PFN-friendly grouping
        self.fc = PartitionedLinear(64, num_classes, num_blocks=2)

        # optional small log
        print(f"  [HourglassBottleneckCNN] bottleneck_width={bottleneck_width} target_width={target_width}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.bottleneck(x))
        x = self.upsample(x)
        x = torch.relu(self.dec1(x))
        x = self.upsample(x)
        x = torch.relu(self.dec2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)
    
    # PFN helper APIs (compatible with other model classes)
    def get_parameter_groups(self) -> List[Dict]:
        """Return parameter groups for optimizer / PFN (per-block grouping)."""
        groups = []
        for layer_name, layer in (('enc1', self.enc1), ('enc2', self.enc2), ('bottleneck', self.bottleneck),
                                 ('dec1', self.dec1), ('dec2', self.dec2)):
            for b in range(layer.num_blocks):
                groups.append({
                    'params': list(layer.blocks[b].parameters()),
                    'name': f'{layer_name}_block{b}',
                    'lr': 0.001
                })
        # classifier head
        groups.append({'params': list(self.fc.parameters()), 'name': 'fc', 'lr': 0.001})
        return groups

    def get_all_block_gradients(self) -> Dict[str, List[torch.Tensor]]:
        """Collect gradients for each block (PFN analysis)."""
        gradients: Dict[str, List[torch.Tensor]] = {}
        device = next(self.parameters()).device
        for layer_name, layer in (('enc1', self.enc1), ('enc2', self.enc2), ('bottleneck', self.bottleneck),
                                 ('dec1', self.dec1), ('dec2', self.dec2)):
            block_grads = []
            for block in layer.blocks:
                grads = [p.grad.flatten() for p in block.parameters() if p.grad is not None]
                block_grads.append(_safe_concat(grads, device))
            gradients[layer_name] = block_grads
        fc_grads = [p.grad.flatten() for p in self.fc.parameters() if p.grad is not None]
        gradients['fc'] = [_safe_concat(fc_grads, device)]
        return gradients

    def get_layer_widths(self) -> Dict[str, int]:
        """Return widths (output channels) per layer for PFN capacity calibration."""
        widths = {}
        for name, layer in (('enc1', self.enc1), ('enc2', self.enc2), ('bottleneck', self.bottleneck),
                            ('dec1', self.dec1), ('dec2', self.dec2)):
            widths[name] = layer.block_size * layer.num_blocks
        widths['fc'] = self.fc.out_features  # PartitionedLinear now has out_features attribute
        return widths


def get_model(scenario: str, dataset: str, **kwargs) -> nn.Module:
    """
    Model factory for PFN experiments.
    支持自动分块模式 (auto_partition)
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
        # return a compact hourglass bottleneck model (encoder -> narrow bottleneck -> decoder)
        b_width = kwargs.get('bottleneck_width', 2)
        return HourglassBottleneckCNN(
            num_classes=num_classes,
            bottleneck_width=b_width,
            target_width=target_width
        )
    raise ValueError(f"Unknown scenario: {scenario}")