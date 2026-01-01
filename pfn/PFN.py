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
    """极简图构建器：容量 = 归一化梯度范数 × 深度补偿"""
    
    def __init__(self, use_hessian_approx: bool = False, depth_penalty: bool = True, history_size: int = 5):
        self.num_nodes = 0
        self.source = 0
        self.sink = 0
        self.node_map = {}
        self.node_names = {}
        self.layer_names = []
        self.current_epoch = 0
        self.total_epochs = 1
        self.history_size = history_size
        self.grad_history: List[Dict[str, List[float]]] = []
        self.debug = True  # 开启调试输出
    
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
    
    def _compute_normalized_energy(self, gradients: Dict[str, List[torch.Tensor]]) -> Dict[str, List[float]]:
        """计算归一化的梯度能量"""
        # 收集所有范数
        all_norms = []
        raw_norms = {}
        
        for layer_name, blocks in gradients.items():
            raw_norms[layer_name] = []
            for grad in blocks:
                norm = grad.norm(2).item()
                raw_norms[layer_name].append(norm)
                all_norms.append(norm)
        
        # 计算平均范数用于归一化
        avg_norm = np.mean(all_norms) + 1e-9
        
        # 归一化
        normalized = {}
        for layer_name, norms in raw_norms.items():
            normalized[layer_name] = [n / avg_norm for n in norms]
        
        return normalized
    
    def _get_smoothed_energy(self, current_energy: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """使用历史平滑能量值，减少噪声"""
        self.grad_history.append(current_energy)
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)
        
        if len(self.grad_history) == 1:
            return current_energy
        
        smoothed = {}
        for layer_name in current_energy:
            smoothed[layer_name] = []
            for b_idx in range(len(current_energy[layer_name])):
                values = [h[layer_name][b_idx] for h in self.grad_history if layer_name in h and b_idx < len(h[layer_name])]
                smoothed[layer_name].append(np.mean(values) if values else current_energy[layer_name][b_idx])
        
        return smoothed
    
    def build_graph(self, gradients: Dict[str, List[torch.Tensor]]) -> Tuple[np.ndarray, Dict]:
        """构建流网络：容量 = 归一化梯度能量 × 深度补偿"""
        self.setup_topology(gradients)
        capacity = np.zeros((self.num_nodes, self.num_nodes))
        
        # 归一化并平滑
        normalized_energy = self._compute_normalized_energy(gradients)
        smoothed_energy = self._get_smoothed_energy(normalized_energy)
        
        num_layers = len(self.layer_names)
        
        for layer_idx, layer_name in enumerate(self.layer_names):
            energies = smoothed_energy[layer_name]
            # 深度补偿：更温和的指数补偿
            depth_scale = 1.0 + 0.2 * layer_idx
            
            for b_idx, energy in enumerate(energies):
                u = self.node_map[(layer_name, b_idx)]
                scaled_energy = max(energy * depth_scale, 0.01)  # 最小容量提高到0.01
                
                # 1. 注入边：只有第一层连Source
                if layer_idx == 0:
                    capacity[self.source][u] = scaled_energy
                
                # 2. 传输边：Mesh Connectivity（交叉连接）
                if layer_idx < num_layers - 1:
                    next_layer = self.layer_names[layer_idx + 1]
                    next_energies = smoothed_energy[next_layer]
                    num_next_blocks = len(next_energies)
                    
                    # 连接到下一层的多个block（mesh connectivity）
                    for offset in [-1, 0, 1]:
                        nb_idx = b_idx + offset
                        if 0 <= nb_idx < num_next_blocks:
                            v = self.node_map[(next_layer, nb_idx)]
                            # 直连权重1.0，邻近权重0.3
                            weight = 1.0 if offset == 0 else 0.3
                            capacity[u][v] = scaled_energy * weight
                
                # 3. 提取边：最后一层连Sink
                if layer_idx == num_layers - 1:
                    capacity[u][self.sink] = scaled_energy
        
        return capacity, {'normalized_energy': smoothed_energy}
    
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
        
        # 计算S集合（可达集）
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
        max_iter = n * n * 2
        
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
    """温和的瓶颈补偿优化器 - 不重置LR，基于当前值调整"""
    
    def __init__(self, optimizer: Optimizer, base_lr: float = 0.001,
                 base_boost: float = 1.3, max_boost: float = 3.0, decay_factor: float = 0.95):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.base_boost = base_boost
        self.max_boost = max_boost
        self.decay_factor = decay_factor
        self.name_to_idx = {g['name']: i for i, g in enumerate(optimizer.param_groups) if 'name' in g}
        self.bottleneck_history: List[List[str]] = []
        self.boost_counts: Dict[str, int] = defaultdict(int)
        self.debug = True
        self.step_count = 0
    
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
        self.step_count += 1
        
        # Debug输出
        if self.debug and self.step_count % 10 == 0:
            total_potential = np.sum(capacity_matrix[graph_builder.source, :])
            print(f"\n[PFN Debug] Step {self.step_count}")
            print(f"  Total Potential Flow: {total_potential:.4f}")
            print(f"  Graph Partition: S={len(S_set)}, T={len(T_set)}")
            
            bottleneck_names = []
            for u, v in cut_edges[:5]:  # 只显示前5个
                u_name = graph_builder.get_node_name(u)
                v_name = graph_builder.get_node_name(v)
                bottleneck_names.append(f"{u_name}->{v_name}")
            print(f"  Cut Edges: {bottleneck_names}")
        
        # 温和衰减所有非BN层的LR（不是重置！）
        for group in self.optimizer.param_groups:
            if 'name' in group and 'bn' not in group['name'].lower():
                # 温和衰减，而不是重置
                current_lr = group['lr']
                target_lr = self.base_lr
                # 缓慢向base_lr回归
                group['lr'] = current_lr * self.decay_factor + target_lr * (1 - self.decay_factor)
        
        # 识别瓶颈节点
        bottleneck_nodes = set()
        for u, v in cut_edges:
            if v != graph_builder.sink:
                bottleneck_nodes.add(v)
            if u != graph_builder.source:
                bottleneck_nodes.add(u)
        
        # 基于当前LR进行boost（不是重置后boost）
        boosted_names = []
        for node_id in bottleneck_nodes:
            node_name = graph_builder.get_node_name(node_id)
            param_name = self._node_to_param_group(node_name)
            
            if param_name and param_name in self.name_to_idx:
                idx = self.name_to_idx[param_name]
                current_lr = self.optimizer.param_groups[idx]['lr']
                
                # 计算boost，考虑flow_deficit
                boost = self.base_boost * (1.0 + min(flow_deficit, 0.5))
                new_lr = min(current_lr * boost, self.base_lr * self.max_boost)
                
                self.optimizer.param_groups[idx]['lr'] = new_lr
                self.boost_counts[param_name] += 1
                boosted_names.append(f"{param_name}({new_lr:.5f})")
        
        if self.debug and self.step_count % 10 == 0 and boosted_names:
            print(f"  Boosted: {boosted_names[:3]}")
        
        self.bottleneck_history.append([f"{graph_builder.get_node_name(u)}->{graph_builder.get_node_name(v)}" for u, v in cut_edges])
    
    def apply_gradient_clipping(self, model, T_set: Set[int], graph_builder):
        pass
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_statistics(self) -> Dict:
        return {
            'bottleneck_history': self.bottleneck_history,
            'boost_counts': dict(self.boost_counts)
        }
