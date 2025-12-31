import torch
from torch.optim import Optimizer
from typing import Dict, List, Set, Tuple
import numpy as np

class BottleneckOptimizer:
    """
    Wrapper that adjusts learning rates based on PFN bottleneck analysis.
    """
    
    def __init__(self, optimizer: Optimizer, base_lr: float = 0.001,
                 boost_factor: float = 1.5, clip_value: float = 1.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.boost_factor = boost_factor
        self.clip_value = clip_value
        
        # Map param group names to indices
        self.name_to_idx = {}
        for idx, group in enumerate(optimizer.param_groups):
            if 'name' in group:
                self.name_to_idx[group['name']] = idx
        
        # Tracking
        self.lr_history: Dict[str, List[float]] = {name: [] for name in self.name_to_idx}
        self.bottleneck_history: List[List[str]] = []
        
    def _node_to_param_group(self, node_idx: int) -> str:
        """Map PFN node index to parameter group name."""
        if node_idx <= 4 and node_idx >= 1:
            return f'layer1_block{node_idx - 1}'
        elif node_idx <= 6 and node_idx >= 5:
            return f'layer2_block{node_idx - 5}'
        elif node_idx == 7:
            return 'layer3_block0'
        return None
    
    def update_learning_rates(self, S_set: Set[int], T_set: Set[int], 
                              cut_edges: List[Tuple[int, int]]):
        """
        Adjust learning rates based on min-cut analysis.
        
        Strategy:
        - Blocks in S_set (before cut): Boost LR to push more gradient flow
        - Blocks in T_set (after cut): Apply gradient clipping for stability
        - Blocks at cut boundary: Moderate boost
        """
        # Reset all LRs to base
        for group in self.optimizer.param_groups:
            group['lr'] = self.base_lr
        
        # Find boundary nodes (in S_set but connected to T_set)
        boundary_nodes = set()
        for u, v in cut_edges:
            if u in S_set:
                boundary_nodes.add(u)
        
        # Boost LR for S_set nodes (especially boundary)
        for node in S_set:
            param_name = self._node_to_param_group(node)
            if param_name and param_name in self.name_to_idx:
                idx = self.name_to_idx[param_name]
                if node in boundary_nodes:
                    self.optimizer.param_groups[idx]['lr'] = self.base_lr * self.boost_factor
                else:
                    self.optimizer.param_groups[idx]['lr'] = self.base_lr * 1.2
        
        # Record for history
        bottleneck_names = []
        for u, v in cut_edges:
            u_name = self._node_to_param_group(u) or f"node{u}"
            v_name = self._node_to_param_group(v) or f"node{v}"
            bottleneck_names.append(f"{u_name}->{v_name}")
        self.bottleneck_history.append(bottleneck_names)
        
        # Record LR history
        for name, idx in self.name_to_idx.items():
            self.lr_history[name].append(self.optimizer.param_groups[idx]['lr'])
    
    def apply_gradient_clipping(self, model: torch.nn.Module, T_set: Set[int]):
        """Apply gradient clipping to T_set blocks for stability."""
        for node in T_set:
            param_name = self._node_to_param_group(node)
            if param_name:
                # Find corresponding parameters and clip
                for group in self.optimizer.param_groups:
                    if group.get('name') == param_name:
                        for p in group['params']:
                            if p.grad is not None:
                                torch.nn.utils.clip_grad_norm_([p], self.clip_value)
    
    def step(self):
        """Perform optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        return {
            'lr_history': self.lr_history,
            'bottleneck_history': self.bottleneck_history,
            'bottleneck_frequency': self._compute_bottleneck_frequency()
        }
    
    def _compute_bottleneck_frequency(self) -> Dict[str, int]:
        """Count how often each edge appears in min-cut."""
        freq = {}
        for bottlenecks in self.bottleneck_history:
            for b in bottlenecks:
                freq[b] = freq.get(b, 0) + 1
        return freq
