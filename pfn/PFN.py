"""
Parameter Flow Network (PFN) - 极简物理版
核心原则：容量 = 梯度能量，不要玄学
"""

import torch
import numpy as np
from torch.optim import Optimizer
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional


class PFNGraphBuilder:
    """极简图构建器：容量 = 梯度范数 × 深度补偿"""
    
    def __init__(self, use_hessian_approx: bool = False, depth_penalty: bool = True, history_size: int = 0):
        self.num_nodes = 0
        self.source = 0
        self.sink = 0
        self.node_map = {}
        self.node_names = {}
        self.layer_names = []
        self.current_epoch = 0
        self.total_epochs = 1
    
    def setup_topology(self, gradients: Dict[str, List[torch.Tensor]]):
        self.layer_names = list(gradients.keys())
        self.node_map = {}
        self.node_names = {}
        
        current_id = 1
        for layer_name in self.layer_names:
            for block_idx in range(len(gradients[layer_name])):
                self.node_map[(layer_name, block_idx)] = current_id
                self.node_names[current_id] = f"{layer_name}_block{block_idx}"
                current_id += 1
        
        self.source = 0
        self.sink = current_id
        self.num_nodes = current_id + 1
    
    def build_graph(self, gradients: Dict[str, List[torch.Tensor]]) -> Tuple[np.ndarray, Dict]:
        """构建流网络：容量 = 梯度能量"""
        self.setup_topology(gradients)
        capacity = np.zeros((self.num_nodes, self.num_nodes))
        
        num_layers = len(self.layer_names)
        
        for layer_idx, layer_name in enumerate(self.layer_names):
            blocks = gradients[layer_name]
            # 深度补偿：深层梯度自然衰减，用exp补偿
            depth_scale = np.exp(0.3 * layer_idx)
            
            for b_idx, grad in enumerate(blocks):
                u = self.node_map[(layer_name, b_idx)]
                energy = grad.norm(2).item() * depth_scale
                
                # 1. 注入边：只有第一层连Source
                if layer_idx == 0:
                    capacity[self.source][u] = max(energy, 1e-6)
                
                # 2. 传输边：并行管道，同Index相连
                if layer_idx < num_layers - 1:
                    next_layer = self.layer_names[layer_idx + 1]
                    next_blocks = gradients[next_layer]
                    # 同Index连接（并行管道）
                    if b_idx < len(next_blocks):
                        v = self.node_map[(next_layer, b_idx)]
                        capacity[u][v] = max(energy, 1e-6)
                    # 额外连接相邻（增加路由灵活性）
                    for offset in [-1, 1]:
                        nb_idx = b_idx + offset
                        if 0 <= nb_idx < len(next_blocks):
                            v = self.node_map[(next_layer, nb_idx)]
                            capacity[u][v] = max(energy * 0.5, 1e-6)
                
                # 3. 提取边：最后一层连Sink
                if layer_idx == num_layers - 1:
                    capacity[u][self.sink] = max(energy, 1e-6)
        
        return capacity, {}
    
    def get_node_name(self, node_idx: int) -> str:
        if node_idx == self.source: return "Source"
        if node_idx == self.sink: return "Sink"
        return self.node_names.get(node_idx, f"Node_{node_idx}")


class IncrementalPushRelabel:
    """简化的Push-Relabel最大流算法"""
    
    def __init__(self):
        self.flow = defaultdict(float)
        self.height = {}
        self.excess = {}
        self.n = 0
    
    def find_min_cut(self, capacity: np.ndarray, source: int, sink: int) -> Tuple[float, List, Set, Set]:
        self.n = capacity.shape[0]
        max_flow = self._solve(capacity, source, sink)
        
        # 计算S集合
        S_set = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in range(self.n):
                if v not in S_set:
                    residual = capacity[u][v] - self.flow.get((u, v), 0)
                    if residual > 1e-9:
                        S_set.add(v)
                        queue.append(v)
        
        T_set = set(range(self.n)) - S_set
        cut_edges = [(u, v) for u in S_set for v in T_set if capacity[u][v] > 1e-9]
        
        return max_flow, cut_edges, S_set, T_set
    
    def _solve(self, capacity: np.ndarray, source: int, sink: int) -> float:
        n = self.n
        self.height = {i: 0 for i in range(n)}
        self.height[source] = n
        self.excess = {i: 0.0 for i in range(n)}
        self.flow = defaultdict(float)
        
        # 预流
        for v in range(n):
            if capacity[source][v] > 1e-9:
                f = capacity[source][v]
                self.flow[(source, v)] = f
                self.flow[(v, source)] = -f
                self.excess[v] = f
                self.excess[source] -= f
        
        active = deque([v for v in range(n) if v != source and v != sink and self.excess[v] > 1e-9])
        max_iter = n * n
        
        for _ in range(max_iter):
            if not active: break
            u = active.popleft()
            if self.excess[u] <= 1e-9: continue
            
            # Push
            for v in range(n):
                if self.excess[u] <= 1e-9: break
                res = capacity[u][v] - self.flow.get((u, v), 0)
                if res > 1e-9 and self.height[u] == self.height[v] + 1:
                    delta = min(self.excess[u], res)
                    self.flow[(u, v)] += delta
                    self.flow[(v, u)] -= delta
                    self.excess[u] -= delta
                    self.excess[v] += delta
                    if v != source and v != sink and self.excess[v] > 1e-9 and v not in active:
                        active.append(v)
            
            # Relabel
            if self.excess[u] > 1e-9:
                min_h = float('inf')
                for v in range(n):
                    if capacity[u][v] - self.flow.get((u, v), 0) > 1e-9:
                        min_h = min(min_h, self.height[v])
                if min_h < float('inf'):
                    self.height[u] = min_h + 1
                    active.append(u)
        
        return max(self.excess.get(sink, 0), 0)


DinicSolver = IncrementalPushRelabel


class BottleneckOptimizer:
    """温和的瓶颈补偿优化器"""
    
    def __init__(self, optimizer: Optimizer, base_lr: float = 0.001,
                 base_boost: float = 1.5, clip_value: float = 5.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.base_boost = base_boost
        self.clip_value = clip_value
        self.name_to_idx = {g['name']: i for i, g in enumerate(optimizer.param_groups) if 'name' in g}
        self.bottleneck_history: List[List[str]] = []
    
    def _node_to_param_group(self, node_name: str) -> Optional[str]:
        if node_name in self.name_to_idx:
            return node_name
        for name in self.name_to_idx:
            if node_name in name:
                return name
        return None
    
    def update_learning_rates(self, S_set: Set[int], T_set: Set[int],
                              cut_edges: List[Tuple[int, int]],
                              capacity_matrix: np.ndarray,
                              graph_builder, flow_deficit: float = 0.0):
        # 重置所有LR（BN除外）
        for group in self.optimizer.param_groups:
            if 'name' in group and 'bn' not in group['name'].lower():
                group['lr'] = self.base_lr
        
        # 识别瓶颈：割边指向的节点
        bottleneck_nodes = {v for u, v in cut_edges if v != graph_builder.sink}
        
        # 温和助推
        for node_id in bottleneck_nodes:
            node_name = graph_builder.get_node_name(node_id)
            param_name = self._node_to_param_group(node_name)
            
            if param_name and param_name in self.name_to_idx:
                idx = self.name_to_idx[param_name]
                boost = self.base_boost * (1.0 + min(flow_deficit, 1.0))
                self.optimizer.param_groups[idx]['lr'] = self.base_lr * boost
        
        self.bottleneck_history.append([f"{graph_builder.get_node_name(u)}->{graph_builder.get_node_name(v)}" for u, v in cut_edges])
    
    def apply_gradient_clipping(self, model, T_set: Set[int], graph_builder):
        # 简化：不做复杂裁剪
        pass
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_statistics(self) -> Dict:
        return {'bottleneck_history': self.bottleneck_history}
