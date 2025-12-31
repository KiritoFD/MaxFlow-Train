import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class PFNGraphBuilder:
    """
    Builds Parameter Flow Network from neural network gradients.
    
    Node mapping:
    - Node 0: Source (S)
    - Nodes 1-4: Layer1 blocks (H1)
    - Nodes 5-6: Layer2 blocks (H2)
    - Node 7: Layer3 block (O)
    - Node 8: Sink (T)
    """
    
    def __init__(self, gradient_history_size: int = 10):
        self.num_nodes = 9  # S + 4 + 2 + 1 + T
        self.source = 0
        self.sink = 8
        self.gradient_history_size = gradient_history_size
        
        # Node indices
        self.layer1_nodes = [1, 2, 3, 4]
        self.layer2_nodes = [5, 6]
        self.layer3_nodes = [7]
        
        # Gradient history for saturation calculation
        self.gradient_history: Dict[str, deque] = {}
        
    def _get_node_index(self, layer: str, block: int) -> int:
        if layer == 'layer1':
            return 1 + block
        elif layer == 'layer2':
            return 5 + block
        elif layer == 'layer3':
            return 7
        raise ValueError(f"Unknown layer: {layer}")
    
    def _compute_gradient_norm(self, grad: torch.Tensor) -> float:
        """L2 norm of gradient."""
        return grad.norm(2).item()
    
    def _compute_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """Cosine similarity between two gradient vectors."""
        # Handle dimension mismatch by using summary statistics
        norm1 = grad1.norm(2)
        norm2 = grad2.norm(2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Use gradient statistics for similarity
        mean1, std1 = grad1.mean(), grad1.std()
        mean2, std2 = grad2.mean(), grad2.std()
        
        # Normalized correlation
        similarity = 1.0 / (1.0 + abs(mean1 - mean2).item() + abs(std1 - std2).item())
        return similarity
    
    def _compute_saturation(self, layer: str, block: int, current_grad: torch.Tensor) -> float:
        """
        Compute saturation based on gradient variance over history.
        Lower variance = higher saturation = lower capacity.
        """
        key = f"{layer}_block{block}"
        
        if key not in self.gradient_history:
            self.gradient_history[key] = deque(maxlen=self.gradient_history_size)
        
        grad_norm = self._compute_gradient_norm(current_grad)
        self.gradient_history[key].append(grad_norm)
        
        if len(self.gradient_history[key]) < 2:
            return 1.0  # Full capacity initially
        
        variance = np.var(list(self.gradient_history[key]))
        # Higher variance = less saturated = higher capacity
        saturation_capacity = 1.0 / (1.0 + np.exp(-variance))
        return saturation_capacity
    
    def build_graph(self, gradients: Dict[str, List[torch.Tensor]]) -> Tuple[np.ndarray, Dict]:
        """
        Build adjacency matrix with capacities from gradients.
        
        Returns:
            capacity_matrix: (num_nodes, num_nodes) capacity matrix
            edge_info: Dictionary with edge metadata
        """
        capacity = np.zeros((self.num_nodes, self.num_nodes))
        edge_info = {'injection': [], 'transfer': [], 'extraction': []}
        
        # 1. Injection edges: S -> Layer1 blocks
        for i, node in enumerate(self.layer1_nodes):
            grad = gradients['layer1'][i]
            cap = self._compute_gradient_norm(grad)
            capacity[self.source][node] = cap
            edge_info['injection'].append((self.source, node, cap))
        
        # 2. Transfer edges: Layer1 -> Layer2
        for i, n1 in enumerate(self.layer1_nodes):
            for j, n2 in enumerate(self.layer2_nodes):
                grad1 = gradients['layer1'][i]
                grad2 = gradients['layer2'][j]
                similarity = self._compute_cosine_similarity(grad1, grad2)
                cap = similarity * (self._compute_gradient_norm(grad1) + 
                                   self._compute_gradient_norm(grad2)) / 2
                capacity[n1][n2] = max(cap, 1e-6)  # Avoid zero capacity
                edge_info['transfer'].append((n1, n2, cap))
        
        # 3. Transfer edges: Layer2 -> Layer3
        for i, n2 in enumerate(self.layer2_nodes):
            for n3 in self.layer3_nodes:
                grad2 = gradients['layer2'][i]
                grad3 = gradients['layer3'][0]
                similarity = self._compute_cosine_similarity(grad2, grad3)
                cap = similarity * (self._compute_gradient_norm(grad2) + 
                                   self._compute_gradient_norm(grad3)) / 2
                capacity[n2][n3] = max(cap, 1e-6)
                edge_info['transfer'].append((n2, n3, cap))
        
        # 4. Extraction edges: Layer3 -> T (based on saturation)
        for i, node in enumerate(self.layer3_nodes):
            grad = gradients['layer3'][i]
            saturation = self._compute_saturation('layer3', i, grad)
            cap = saturation * self._compute_gradient_norm(grad)
            capacity[node][self.sink] = max(cap, 1e-6)
            edge_info['extraction'].append((node, self.sink, cap))
        
        # Also add extraction from layer2 to sink (for completeness)
        for i, node in enumerate(self.layer2_nodes):
            grad = gradients['layer2'][i]
            saturation = self._compute_saturation('layer2', i, grad)
            cap = saturation * self._compute_gradient_norm(grad) * 0.5
            capacity[node][self.sink] = max(cap, 1e-6)
            edge_info['extraction'].append((node, self.sink, cap))
        
        return capacity, edge_info
    
    def get_node_name(self, node_idx: int) -> str:
        """Get human-readable node name."""
        if node_idx == 0:
            return "Source"
        elif node_idx == 8:
            return "Sink"
        elif node_idx <= 4:
            return f"L1_B{node_idx-1}"
        elif node_idx <= 6:
            return f"L2_B{node_idx-5}"
        else:
            return "L3_B0"
