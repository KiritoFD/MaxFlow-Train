"""
Parameter Flow Network (PFN) - 联合优化版
核心改进：基于排名的容量系统 (Rank-Based Capacity)
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
from torch.optim import Optimizer
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional


class PFNLogger:
    """PFN专用日志记录器"""
    
    def __init__(self, log_dir: str = './pfn_logs', enabled: bool = True):
        self.enabled = enabled
        self.log_dir = log_dir
        self.log_file = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_logs: List[Dict] = []
        self.summary_stats = {
            'total_steps': 0,
            'bottleneck_frequency': defaultdict(int),
            'lr_history': defaultdict(list),
            'flow_deficit_history': [],
            'max_flow_history': []
        }
        
        if enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f'pfn_{self.session_id}.log')
            self._write_header()
    
    def _write_header(self):
        """写入日志头"""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"PFN Debug Log - Session {self.session_id}\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_step(self, step: int, epoch: int, data: Dict):
        """记录每个PFN步骤的详细信息"""
        if not self.enabled:
            return
        
        self.summary_stats['total_steps'] += 1
        
        # 记录到内存
        log_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        self.step_logs.append(log_entry)
        
        # 更新统计
        if 'flow_deficit' in data:
            self.summary_stats['flow_deficit_history'].append(data['flow_deficit'])
        if 'max_flow' in data:
            self.summary_stats['max_flow_history'].append(data['max_flow'])
        if 'bottlenecks' in data:
            for bn in data['bottlenecks']:
                self.summary_stats['bottleneck_frequency'][bn] += 1
        
        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(f"\n[Step {step}] Epoch {epoch}\n")
            f.write("-" * 40 + "\n")
            
            if 'graph_info' in data:
                gi = data['graph_info']
                f.write(f"  Graph Topology:\n")
                f.write(f"    - Nodes: {gi.get('num_nodes', 'N/A')}\n")
                f.write(f"    - Layers: {gi.get('num_layers', 'N/A')}\n")
                f.write(f"    - Blocks per layer: {gi.get('blocks_per_layer', 'N/A')}\n")
            
            if 'flow_info' in data:
                fi = data['flow_info']
                f.write(f"  Flow Analysis:\n")
                f.write(f"    - Total Potential: {fi.get('total_potential', 0):.4f}\n")
                f.write(f"    - Max Flow: {fi.get('max_flow', 0):.4f}\n")
                f.write(f"    - Flow Deficit: {fi.get('flow_deficit', 0):.4f}\n")
                f.write(f"    - S set size: {fi.get('s_size', 0)}\n")
                f.write(f"    - T set size: {fi.get('t_size', 0)}\n")
            
            if 'cut_edges' in data:
                f.write(f"  Cut Edges (Top 10):\n")
                for i, edge in enumerate(data['cut_edges'][:10]):
                    f.write(f"    {i+1}. {edge}\n")
            
            if 'bottlenecks' in data:
                f.write(f"  Identified Bottlenecks:\n")
                for bn in data['bottlenecks'][:10]:
                    f.write(f"    - {bn}\n")
            
            if 'lr_updates' in data:
                f.write(f"  LR Updates:\n")
                for name, lr_info in list(data['lr_updates'].items())[:10]:
                    f.write(f"    - {name}: {lr_info['old']:.6f} -> {lr_info['new']:.6f}")
                    if lr_info.get('boosted'):
                        f.write(f" [BOOSTED x{lr_info.get('boost_factor', 1):.2f}]")
                    f.write("\n")
            
            if 'rank_energy' in data:
                f.write(f"  Rank Energy (per layer):\n")
                for layer, energies in data['rank_energy'].items():
                    avg_e = np.mean(energies) if energies else 0
                    min_e = np.min(energies) if energies else 0
                    max_e = np.max(energies) if energies else 0
                    f.write(f"    - {layer}: avg={avg_e:.3f}, min={min_e:.3f}, max={max_e:.3f}, blocks={len(energies)}\n")
    
    def log_epoch_summary(self, epoch: int, train_loss: float, test_acc: float):
        """记录epoch摘要"""
        if not self.enabled:
            return
        
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"EPOCH {epoch} SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Train Loss: {train_loss:.4f}\n")
            f.write(f"  Test Accuracy: {test_acc:.4f}\n")
            
            # 统计本epoch的瓶颈频率
            recent_logs = [l for l in self.step_logs if l.get('epoch') == epoch]
            if recent_logs:
                bn_counts = defaultdict(int)
                for log in recent_logs:
                    for bn in log.get('bottlenecks', []):
                        bn_counts[bn] += 1
                
                if bn_counts:
                    f.write(f"  Top Bottlenecks this Epoch:\n")
                    sorted_bns = sorted(bn_counts.items(), key=lambda x: -x[1])[:5]
                    for bn, count in sorted_bns:
                        f.write(f"    - {bn}: {count} times\n")
    
    def save_summary(self):
        """保存完整的统计摘要"""
        if not self.enabled:
            return
        
        summary_file = os.path.join(self.log_dir, f'pfn_summary_{self.session_id}.json')
        
        # 转换defaultdict为普通dict
        summary = {
            'session_id': self.session_id,
            'total_steps': self.summary_stats['total_steps'],
            'bottleneck_frequency': dict(self.summary_stats['bottleneck_frequency']),
            'flow_deficit_stats': {
                'mean': float(np.mean(self.summary_stats['flow_deficit_history'])) if self.summary_stats['flow_deficit_history'] else 0,
                'std': float(np.std(self.summary_stats['flow_deficit_history'])) if self.summary_stats['flow_deficit_history'] else 0,
                'max': float(np.max(self.summary_stats['flow_deficit_history'])) if self.summary_stats['flow_deficit_history'] else 0,
            },
            'max_flow_stats': {
                'mean': float(np.mean(self.summary_stats['max_flow_history'])) if self.summary_stats['max_flow_history'] else 0,
                'final': float(self.summary_stats['max_flow_history'][-1]) if self.summary_stats['max_flow_history'] else 0,
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 写入日志文件结尾
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SESSION COMPLETE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total PFN Steps: {summary['total_steps']}\n")
            f.write(f"Flow Deficit - Mean: {summary['flow_deficit_stats']['mean']:.4f}, Max: {summary['flow_deficit_stats']['max']:.4f}\n")
            
            # Top 10 最频繁的瓶颈
            if summary['bottleneck_frequency']:
                f.write("\nTop 10 Most Frequent Bottlenecks:\n")
                sorted_bns = sorted(summary['bottleneck_frequency'].items(), key=lambda x: -x[1])[:10]
                for bn, count in sorted_bns:
                    f.write(f"  {bn}: {count} times\n")
        
        print(f"[PFN] Summary saved to {summary_file}")
        return summary_file


# 全局logger实例
_pfn_logger: Optional[PFNLogger] = None

def get_pfn_logger(log_dir: str = './pfn_logs', enabled: bool = True) -> PFNLogger:
    """获取或创建PFN日志记录器"""
    global _pfn_logger
    if _pfn_logger is None:
        _pfn_logger = PFNLogger(log_dir, enabled)
    return _pfn_logger

def reset_pfn_logger(log_dir: str = './pfn_logs', enabled: bool = True) -> PFNLogger:
    """重置PFN日志记录器（用于新实验）"""
    global _pfn_logger
    _pfn_logger = PFNLogger(log_dir, enabled)
    return _pfn_logger


class PFNGraphBuilder:
    """联合优化版图构建器：排名容量 + 动态连接窗口"""
    
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
        self.debug = True
        self.enable_skip_connections = True
        self.skip_distance = 2
        self.logger: Optional[PFNLogger] = None
    
    def set_logger(self, logger: PFNLogger):
        """设置日志记录器"""
        self.logger = logger
    
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
    
    def _compute_rank_based_energy(self, gradients: Dict[str, List[torch.Tensor]]) -> Dict[str, List[float]]:
        """
        联合优化核心：基于排名的相对能量计算
        彻底解决深度网络梯度消失导致的容量失衡问题
        """
        normalized = {}
        
        for layer_name, blocks in gradients.items():
            num_blocks = len(blocks)
            
            # 1. 计算原始模长
            raw_norms = np.array([g.norm(2).item() for g in blocks])
            
            # 2. 如果全层梯度极小（死层），给予最低容量
            if np.max(raw_norms) < 1e-9:
                normalized[layer_name] = [0.1] * num_blocks
                continue
            
            # 3. 计算相对排名 (0.0 ~ 1.0)
            if num_blocks == 1:
                normalized[layer_name] = [1.0]
            else:
                # argsort两次得到排名
                ranks = np.argsort(np.argsort(raw_norms)).astype(float)
                max_rank = num_blocks - 1
                # 线性映射到 [0.1, 1.0] 区间，保证最弱的也有底限
                rank_scores = 0.1 + 0.9 * (ranks / max_rank)
                normalized[layer_name] = rank_scores.tolist()
        
        return normalized
    
    def _get_smoothed_energy(self, current_energy: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """使用指数移动平均平滑能量值"""
        self.grad_history.append(current_energy)
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)
        
        if len(self.grad_history) == 1:
            return current_energy
        
        weights = np.exp(np.linspace(-1, 0, len(self.grad_history)))
        weights /= weights.sum()
        
        smoothed = {}
        for layer_name in current_energy:
            smoothed[layer_name] = []
            for b_idx in range(len(current_energy[layer_name])):
                values = []
                for h in self.grad_history:
                    if layer_name in h and b_idx < len(h[layer_name]):
                        values.append(h[layer_name][b_idx])
                
                if len(values) == len(weights):
                    smoothed[layer_name].append(np.dot(weights, values))
                else:
                    smoothed[layer_name].append(current_energy[layer_name][b_idx])
        
        return smoothed
    
    def build_graph(self, gradients: Dict[str, List[torch.Tensor]]) -> Tuple[np.ndarray, Dict]:
        """构建流网络：排名容量 + 动态连接窗口"""
        self.setup_topology(gradients)
        capacity = np.zeros((self.num_nodes, self.num_nodes))
        
        # === 使用排名能量替代绝对范数 ===
        rank_energy = self._compute_rank_based_energy(gradients)
        smoothed_energy = self._get_smoothed_energy(rank_energy)
        
        num_layers = len(self.layer_names)
        
        # 记录每层的分块数（用于日志）
        blocks_per_layer = {name: len(smoothed_energy[name]) for name in self.layer_names}
        
        for layer_idx, layer_name in enumerate(self.layer_names):
            energies = smoothed_energy[layer_name]
            num_curr_blocks = len(energies)
            
            for b_idx, energy in enumerate(energies):
                u = self.node_map[(layer_name, b_idx)]
                cap_val = energy
                
                if layer_idx == 0:
                    capacity[self.source][u] = cap_val
                
                if layer_idx < num_layers - 1:
                    next_layer = self.layer_names[layer_idx + 1]
                    next_energies = smoothed_energy[next_layer]
                    num_next_blocks = len(next_energies)
                    
                    ratio = num_next_blocks / num_curr_blocks
                    window = max(1, int(np.ceil(ratio))) + 1
                    
                    center_next = int(b_idx * ratio)
                    start_next = max(0, center_next - 1)
                    end_next = min(num_next_blocks, center_next + window)
                    
                    for nb_idx in range(start_next, end_next):
                        v = self.node_map[(next_layer, nb_idx)]
                        edge_cap = next_energies[nb_idx]
                        weight = 1.0 if nb_idx == center_next else 0.5
                        capacity[u][v] = edge_cap * weight
                
                if self.enable_skip_connections and layer_idx < num_layers - self.skip_distance:
                    skip_layer = self.layer_names[layer_idx + self.skip_distance]
                    skip_energies = smoothed_energy[skip_layer]
                    num_skip_blocks = len(skip_energies)
                    
                    skip_ratio = num_skip_blocks / num_curr_blocks
                    target_idx = min(int(b_idx * skip_ratio), num_skip_blocks - 1)
                    
                    v = self.node_map[(skip_layer, target_idx)]
                    skip_cap = skip_energies[target_idx] * 0.2
                    capacity[u][v] = max(skip_cap, 0.05)
                
                if layer_idx == num_layers - 1:
                    capacity[u][self.sink] = cap_val
        
        metadata = {
            'rank_energy': smoothed_energy,
            'blocks_per_layer': blocks_per_layer,
            'num_nodes': self.num_nodes,
            'num_layers': num_layers
        }
        
        return capacity, metadata
    
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
    """联合优化版瓶颈优化器：支持动态分块数"""
    
    def __init__(self, optimizer: Optimizer, base_lr: float = 0.001,
                 base_boost: float = 1.5, max_boost: float = 4.0, decay_factor: float = 0.92,
                 lr_momentum: float = 0.8):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.base_boost = base_boost
        self.max_boost = max_boost
        self.decay_factor = decay_factor
        self.lr_momentum = lr_momentum
        self.name_to_idx = {g['name']: i for i, g in enumerate(optimizer.param_groups) if 'name' in g}
        self.bottleneck_history: List[List[str]] = []
        self.boost_counts: Dict[str, int] = defaultdict(int)
        self.debug = True
        self.step_count = 0
        self.target_lrs: Dict[str, float] = {name: base_lr for name in self.name_to_idx}
        self.persistent_bottleneck: Dict[str, int] = defaultdict(int)
        self.logger: Optional[PFNLogger] = None
        self.current_epoch = 0
    
    def set_logger(self, logger: PFNLogger):
        """设置日志记录器"""
        self.logger = logger
    
    def update_learning_rates(self, S_set: Set[int], T_set: Set[int],
                              cut_edges: List[Tuple[int, int]],
                              capacity_matrix: np.ndarray,
                              graph_builder, flow_deficit: float = 0.0):
        self.step_count += 1
        
        total_potential = np.sum(capacity_matrix[graph_builder.source, :])
        max_flow = total_potential * (1 - flow_deficit)
        
        # 准备日志数据
        log_data = {
            'graph_info': {
                'num_nodes': graph_builder.num_nodes,
                'num_layers': len(graph_builder.layer_names),
                'blocks_per_layer': {name: len([k for k in graph_builder.node_map if k[0] == name]) 
                                     for name in graph_builder.layer_names}
            },
            'flow_info': {
                'total_potential': float(total_potential),
                'max_flow': float(max_flow),
                'flow_deficit': float(flow_deficit),
                's_size': len(S_set),
                't_size': len(T_set)
            },
            'cut_edges': [],
            'bottlenecks': [],
            'lr_updates': {},
            'max_flow': float(max_flow),
            'flow_deficit': float(flow_deficit)
        }
        
        # 记录cut edges
        for u, v in cut_edges[:20]:
            u_name = graph_builder.get_node_name(u)
            v_name = graph_builder.get_node_name(v)
            cap = capacity_matrix[u][v]
            log_data['cut_edges'].append(f"{u_name} -> {v_name} (cap={cap:.3f})")
        
        # Debug输出（保留原有的控制台输出）
        if self.debug and self.step_count % 10 == 0:
            print(f"\n[PFN Debug] Step {self.step_count}")
            print(f"  Total Potential Flow: {total_potential:.4f}, Flow Deficit: {flow_deficit:.4f}")
            print(f"  Graph: {len(S_set)} in S, {len(T_set)} in T, {len(cut_edges)} cut edges")
            
            bottleneck_names = []
            for u, v in cut_edges[:3]:
                u_name = graph_builder.get_node_name(u)
                v_name = graph_builder.get_node_name(v)
                cap = capacity_matrix[u][v]
                bottleneck_names.append(f"{u_name}->{v_name}({cap:.2f})")
            if bottleneck_names:
                print(f"  Top Bottlenecks: {bottleneck_names}")
        
        # 识别当前瓶颈节点
        current_bottlenecks = set()
        for u, v in cut_edges:
            if v != graph_builder.sink:
                node_name = graph_builder.get_node_name(v)
                param_name = self._node_to_param_group(node_name)
                if param_name:
                    current_bottlenecks.add(param_name)
                    log_data['bottlenecks'].append(param_name)
            if u != graph_builder.source:
                node_name = graph_builder.get_node_name(u)
                param_name = self._node_to_param_group(node_name)
                if param_name:
                    current_bottlenecks.add(param_name)
                    if param_name not in log_data['bottlenecks']:
                        log_data['bottlenecks'].append(param_name)
        
        # 更新持续瓶颈计数
        for name in self.name_to_idx:
            if name in current_bottlenecks:
                self.persistent_bottleneck[name] += 1
            else:
                self.persistent_bottleneck[name] = max(0, self.persistent_bottleneck[name] - 1)
        
        # 计算目标LR
        boosted_names = []
        for name in self.name_to_idx:
            if 'bn' in name.lower():
                continue
            
            idx = self.name_to_idx[name]
            old_lr = self.optimizer.param_groups[idx]['lr']
            
            if name in current_bottlenecks:
                persistence = min(self.persistent_bottleneck[name], 10)
                dynamic_boost = self.base_boost * (1.0 + 0.1 * persistence)
                dynamic_boost = min(dynamic_boost, self.max_boost)
                
                target = self.base_lr * dynamic_boost * (1.0 + min(flow_deficit, 0.5))
                target = min(target, self.base_lr * self.max_boost)
                
                self.target_lrs[name] = target
                self.boost_counts[name] += 1
                boosted_names.append(f"{name[:15]}({target:.5f})")
                
                log_data['lr_updates'][name] = {
                    'old': float(old_lr),
                    'new': float(target),
                    'boosted': True,
                    'boost_factor': float(dynamic_boost),
                    'persistence': persistence
                }
            else:
                current_target = self.target_lrs.get(name, self.base_lr)
                new_target = current_target * self.decay_factor + self.base_lr * (1 - self.decay_factor)
                self.target_lrs[name] = new_target
                
                log_data['lr_updates'][name] = {
                    'old': float(old_lr),
                    'new': float(new_target),
                    'boosted': False
                }
        
        # 平滑LR变化
        for name, idx in self.name_to_idx.items():
            if 'bn' in name.lower():
                continue
            current_lr = self.optimizer.param_groups[idx]['lr']
            target_lr = self.target_lrs.get(name, self.base_lr)
            new_lr = self.lr_momentum * current_lr + (1 - self.lr_momentum) * target_lr
            self.optimizer.param_groups[idx]['lr'] = new_lr
            
            # 更新日志中的实际新LR
            if name in log_data['lr_updates']:
                log_data['lr_updates'][name]['actual_new'] = float(new_lr)
        
        if self.debug and self.step_count % 10 == 0 and boosted_names:
            print(f"  Boosted ({len(boosted_names)}): {boosted_names[:3]}...")
        
        # 记录到日志
        if self.logger:
            self.logger.log_step(self.step_count, self.current_epoch, log_data)
        
        self.bottleneck_history.append([
            f"{graph_builder.get_node_name(u)}->{graph_builder.get_node_name(v)}" 
            for u, v in cut_edges[:5]
        ])
    
    def apply_gradient_clipping(self, model, T_set: Set[int], graph_builder):
        pass
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_statistics(self) -> Dict:
        return {
            'bottleneck_history': self.bottleneck_history[-100:],  # 只保留最近100条
            'boost_counts': dict(self.boost_counts),
            'persistent_bottleneck': dict(self.persistent_bottleneck)
        }