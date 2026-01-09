#%%writefile pfn.py
"""
Parameter Flow Network (PFN) - 联合优化版
核心改进：基于排名的容量系统 (Rank-Based Capacity) + 动态连接范围
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from torch.optim import Optimizer
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional


class PFNLogger:
    """PFN流和最小割的日志记录器"""
    
    def __init__(self, log_dir: str = './pfn_log'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self.epoch_logs = defaultdict(lambda: {'critical_cuts': [], 'steps_count': 0})
        self.metadata = {
            'start_time': self.timestamp,
            'total_epochs': 0
        }
        self.bottleneck_evolution = defaultdict(lambda: {'first_epoch': None, 'last_epoch': None, 'count': 0})
    
    def log_step(self, step: int, epoch: int, cut_edges: List[Tuple[int, int]],
                 flow_dict: Dict[Tuple[int, int], float], capacity_matrix: np.ndarray,
                 S_set: Set[int], T_set: Set[int], graph_builder, max_flow: float,
                 flow_deficit: float = 0.0):
        """记录单步的最小割信息，按epoch聚合"""
        
        # 找出关键割（容量最小的割边）
        if not cut_edges:
            return
        
        cut_capacities = []
        cut_info = []
        
        for u, v in cut_edges:
            u_name = graph_builder.get_node_name(u)
            v_name = graph_builder.get_node_name(v)
            cap = float(capacity_matrix[u][v]) if u < capacity_matrix.shape[0] and v < capacity_matrix.shape[1] else 0.0
            
            cut_capacities.append(cap)
            cut_info.append({
                'from': u_name,
                'to': v_name,
                'capacity': cap,
                'is_source': (u == graph_builder.source),
                'is_sink': (v == graph_builder.sink)
            })
        
        # 找出最小容量的关键割
        if cut_capacities:
            min_cap = min(cut_capacities)
            critical_indices = [i for i, cap in enumerate(cut_capacities) if cap <= min_cap * 1.01]  # 允许1%误差
            
            # 记录关键割信息
            for idx in critical_indices:
                edge = cut_info[idx]
                critical_cut = {
                    'from': edge['from'],
                    'to': edge['to'],
                    'capacity': edge['capacity'],
                    'step': step,
                    'flow': float(flow_dict.get((edge['from_id'], edge['to_id']), 0.0)) if 'from_id' in cut_info[idx] else 0.0
                }
                self.epoch_logs[epoch]['critical_cuts'].append(critical_cut)
        
        self.epoch_logs[epoch]['steps_count'] += 1
        
        # 更新瓶颈节点演变
        for edge in cut_info:
            node_name = edge['to']
            if node_name != "Sink" and not edge['is_sink']:
                if self.bottleneck_evolution[node_name]['first_epoch'] is None:
                    self.bottleneck_evolution[node_name]['first_epoch'] = epoch
                self.bottleneck_evolution[node_name]['last_epoch'] = epoch
                self.bottleneck_evolution[node_name]['count'] += 1
    
    def save_logs(self, dataset: str = '', scenario: str = ''):
        """保存按epoch聚合的日志"""
        if not self.epoch_logs:
            print(f"  [Log] No logs to save")
            return
        
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_epochs'] = len(self.epoch_logs)
        self.metadata['dataset'] = dataset
        self.metadata['scenario'] = scenario
        
        # 按epoch组织日志
        epochs_data = {}
        for epoch in sorted(self.epoch_logs.keys()):
            epoch_data = self.epoch_logs[epoch]
            
            # 统计该epoch的关键割
            critical_cuts = epoch_data['critical_cuts']
            cut_summary = {}
            for cut in critical_cuts:
                key = f"{cut['from']}->{cut['to']}"
                if key not in cut_summary:
                    cut_summary[key] = {
                        'from': cut['from'],
                        'to': cut['to'],
                        'capacity': cut['capacity'],
                        'count': 0,
                        'steps': []
                    }
                cut_summary[key]['count'] += 1
                cut_summary[key]['steps'].append(cut['step'])
            
            # 按出现频率排序
            sorted_cuts = sorted(cut_summary.items(), key=lambda x: x[1]['count'], reverse=True)
            
            epochs_data[f"epoch_{epoch}"] = {
                'critical_cuts': [cut for _, cut in sorted_cuts],
                'total_steps': epoch_data['steps_count'],
                'unique_bottlenecks': len(cut_summary)
            }
        
        output = {
            'metadata': self.metadata,
            'epochs': epochs_data
        }
        
        log_file = os.path.join(self.run_dir, f'pfn_logs_{dataset}_{scenario}.json')
        with open(log_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # 保存瓶颈演变报告
        self._save_bottleneck_report(dataset, scenario)
        
        print(f"  [Log] Saved to {log_file}")
    
    def _save_bottleneck_report(self, dataset: str = '', scenario: str = ''):
        """生成按epoch的瓶颈演变报告"""
        if not self.bottleneck_evolution:
            return
        
        # 按出现频率排序
        bottleneck_list = sorted(
            self.bottleneck_evolution.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        report = {
            'dataset': dataset,
            'scenario': scenario,
            'total_unique_bottlenecks': len(bottleneck_list),
            'bottlenecks': [
                {
                    'node': name,
                    'appearance_count': info['count'],
                    'first_epoch': info['first_epoch'],
                    'last_epoch': info['last_epoch'],
                    'epoch_span': info['last_epoch'] - info['first_epoch'] + 1,
                    'persistence': f"{info['count'] / (info['last_epoch'] - info['first_epoch'] + 1) * 100:.1f}%"
                }
                for name, info in bottleneck_list[:20]
            ]
        }
        
        report_file = os.path.join(self.run_dir, f'bottleneck_report_{dataset}_{scenario}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  [Bottleneck Report] Saved to {report_file}")
    
    def get_run_dir(self) -> str:
        """获取当前运行的日志目录"""
        return self.run_dir


class PFNGraphBuilder:
    """联合优化版图构建器：排名容量 + 动态连接"""
    
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
        # 修改方案2：降低动量，提高灵敏度
        self.ema_alpha = 0.3  # 从原来的隐式0.9降到0.5，2-3步即可反映变化
        self.layer_widths: Dict[str, int] = {}  # 新增：记录各层宽度
    
    def setup_topology(self, gradients: Dict[str, List[torch.Tensor]], layer_widths: Optional[Dict[str, int]] = None):
        self.layer_names = list(gradients.keys())
        self.node_map = {}
        self.node_names = {}
        
        # 记录层宽度信息
        if layer_widths:
            self.layer_widths = layer_widths
        
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
            
            # 2. 处理极端情况：全层梯度极小（死层）
            if np.max(raw_norms) < 1e-12:
                normalized[layer_name] = [0.1] * num_blocks
                continue
            
            # 3. 单块情况
            if num_blocks == 1:
                normalized[layer_name] = [1.0]
                continue
            
            # 4. 计算相对排名 (0.0 ~ 1.0)
            # argsort两次得到排名：[0.1, 0.5, 0.2] -> rank [0, 2, 1]
            ranks = np.argsort(np.argsort(raw_norms)).astype(float)
            max_rank = num_blocks - 1
            
            # 线性映射到 [0.1, 1.0] 区间
            # 最弱的也有0.1的底限，防止断流
            rank_scores = 0.05 + 0.95 * (ranks / max_rank)
            normalized[layer_name] = rank_scores.tolist()
        
        return normalized
    
    def _get_smoothed_energy(self, current_energy: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """使用快速响应的EMA平滑能量值（修改方案2：提高灵敏度）"""
        self.grad_history.append(current_energy)
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)
        
        if len(self.grad_history) == 1:
            return current_energy
        
        # 修改：使用更激进的权重，让最新值占主导
        # 原来是exp(-1,0)约等于(0.37, 1.0)，现在改为更快响应
        num_hist = len(self.grad_history)
        weights = np.array([self.ema_alpha ** (num_hist - 1 - i) for i in range(num_hist)])
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
    
    def build_graph(self, gradients: Dict[str, List[torch.Tensor]], layer_widths: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Dict]:
        """构建流网络：排名容量 + 动态连接范围 + 宽度校正（修改方案3）"""
        self.setup_topology(gradients, layer_widths)
        capacity = np.zeros((self.num_nodes, self.num_nodes))
        
        rank_energy = self._compute_rank_based_energy(gradients)
        smoothed_energy = self._get_smoothed_energy(rank_energy)
        
        num_layers = len(self.layer_names)
        
        # 计算各层的宽度因子（用于Sink连接校正）
        width_factors = {}
        if self.layer_widths:
            max_width = max(self.layer_widths.values()) if self.layer_widths else 1
            for name, width in self.layer_widths.items():
                # 宽度因子：窄层获得更高的Sink容量，防止成为固定瓶颈
                width_factors[name] = np.sqrt(max_width / (width + 1e-6))
        
        for layer_idx, layer_name in enumerate(self.layer_names):
            energies = smoothed_energy[layer_name]
            num_blocks = len(energies)
            
            # 获取该层的宽度因子
            layer_width_factor = width_factors.get(layer_name, 1.0)
            
            for b_idx, energy in enumerate(energies):
                u = self.node_map[(layer_name, b_idx)]
                cap_val = energy
                
                # 1. Source -> 第一层
                if layer_idx == 0:
                    capacity[self.source][u] = cap_val
                
                # 2. 层间连接：动态连接范围
                if layer_idx < num_layers - 1:
                    next_layer = self.layer_names[layer_idx + 1]
                    next_energies = smoothed_energy[next_layer]
                    num_next_blocks = len(next_energies)
                    
                    ratio = num_next_blocks / num_blocks
                    window = max(1, int(np.ceil(ratio))) + 1
                    
                    center_next = int(b_idx * ratio)
                    start_next = max(0, center_next - 1)
                    end_next = min(num_next_blocks, center_next + window)
                    
                    for nb_idx in range(start_next, end_next):
                        v = self.node_map[(next_layer, nb_idx)]
                        weight = 1.0 if nb_idx == center_next else 0.5
                        capacity[u][v] = next_energies[nb_idx] * weight
                
                # 3. 跨层连接（虚拟残差）
                if self.enable_skip_connections and layer_idx < num_layers - self.skip_distance:
                    skip_layer = self.layer_names[layer_idx + self.skip_distance]
                    skip_energies = smoothed_energy[skip_layer]
                    num_skip_blocks = len(skip_energies)
                    
                    skip_ratio = num_skip_blocks / num_blocks
                    skip_idx = min(int(b_idx * skip_ratio), num_skip_blocks - 1)
                    
                    v = self.node_map[(skip_layer, skip_idx)]
                    skip_cap = cap_val * 0.15
                    capacity[u][v] = max(skip_cap, 0.02)
                
                # 4. 最后一层 -> Sink（修改方案3：宽度校正）
                if layer_idx == num_layers - 1:
                    # 原来：capacity[u][self.sink] = cap_val
                    # 修改：窄层（如FC）获得更高的Sink容量，防止固定瓶颈
                    sink_cap = cap_val * layer_width_factor
                    capacity[u][self.sink] = sink_cap
        
        return capacity, {'rank_energy': smoothed_energy, 'width_factors': width_factors}
    
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
    """联合优化版瓶颈优化器：累积补偿机制（修改方案1）"""
    
    def __init__(self, optimizer: Optimizer, base_lr: float = 0.001,
                 base_boost: float = 1.8, max_boost: float = 20.0, decay_factor: float = 0.90,
                 lr_momentum: float = 0.75, logger: Optional['PFNLogger'] = None,
                 accumulated_boost: float = 1.3):  # 新增：累积增益系数
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
        self.logger = logger
        # 修改方案1：累积补偿
        self.accumulated_boost = accumulated_boost
        self.max_accumulated_boost = 40.0  # 累积boost上限
    
    def _node_to_param_group(self, node_name: str) -> Optional[str]:
        # 精确匹配
        if node_name in self.name_to_idx:
            return node_name
        # 模糊匹配：处理 "layer0_block0" -> "L0_B0" 的映射
        for name in self.name_to_idx:
            # 提取layer和block编号进行匹配
            if node_name.replace('layer', 'L').replace('_block', '_B') == name:
                return name
            if node_name.replace('conv', 'conv').replace('_block', '_block') == name:
                return name
            if node_name in name or name in node_name:
                return name
        return None
    
    def update_learning_rates(self, S_set: Set[int], T_set: Set[int],
                              cut_edges: List[Tuple[int, int]],
                              capacity_matrix: np.ndarray,
                              graph_builder, flow_deficit: float = 0.0,
                              flow_dict: Optional[Dict] = None, max_flow: float = 0.0,
                              epoch: int = 0):
        self.step_count += 1
        
        # 记录日志
        if self.logger and flow_dict is not None:
            self.logger.log_step(self.step_count, epoch, cut_edges, flow_dict,
                                capacity_matrix, S_set, T_set, graph_builder,
                                max_flow, flow_deficit)
        
        # Debug输出
        if self.debug and self.step_count % 10 == 0:
            total_potential = np.sum(capacity_matrix[graph_builder.source, :])
            print(f"\n[PFN-Joint] Step {self.step_count}")
            print(f"  Potential: {total_potential:.4f} | S={len(S_set)} T={len(T_set)}")
            
            bottleneck_names = []
            for u, v in cut_edges[:5]:
                u_name = graph_builder.get_node_name(u)
                v_name = graph_builder.get_node_name(v)
                bottleneck_names.append(f"{u_name}->{v_name}")
            if bottleneck_names:
                print(f"  Cuts: {bottleneck_names}")
        
        # 识别当前瓶颈节点
        current_bottlenecks = set()
        for u, v in cut_edges:
            if v != graph_builder.sink:
                node_name = graph_builder.get_node_name(v)
                param_name = self._node_to_param_group(node_name)
                if param_name:
                    current_bottlenecks.add(param_name)
            if u != graph_builder.source:
                node_name = graph_builder.get_node_name(u)
                param_name = self._node_to_param_group(node_name)
                if param_name:
                    current_bottlenecks.add(param_name)
        
        # 修改方案1：累积补偿机制
        boosted_names = []
        for name in self.name_to_idx:
            if 'bn' in name.lower():
                continue
            
            if name in current_bottlenecks:
                # 累积计数
                self.persistent_bottleneck[name] += 1
                persistence = self.persistent_bottleneck[name]
                
                # 核心修改：指数级累积增益
                # dynamic_boost = base_boost * (accumulated_boost ^ (persistence - 1))
                accumulated_factor = self.accumulated_boost ** (persistence - 1)
                accumulated_factor = min(accumulated_factor, self.max_accumulated_boost / self.base_boost)
                
                dynamic_boost = self.base_boost * accumulated_factor
                dynamic_boost = min(dynamic_boost, self.max_boost * (1 + persistence * 0.1))  # 允许突破max_boost
                
                # 额外考虑flow_deficit
                target = self.base_lr * dynamic_boost * (1.0 + min(flow_deficit, 0.8))
                
                self.target_lrs[name] = target
                self.boost_counts[name] += 1
                boosted_names.append(f"{name[:12]}(p={persistence},b={dynamic_boost:.1f})")
            else:
                # 离开最小割：重置计数，缓慢恢复
                self.persistent_bottleneck[name] = 0
                current_target = self.target_lrs.get(name, self.base_lr)
                # 更快的衰减，让资源流向新瓶颈
                self.target_lrs[name] = current_target * 0.85 + self.base_lr * 0.15
        
        # 使用动量平滑实际LR变化（降低动量以更快响应)
        effective_momentum = max(0.5, self.lr_momentum - 0.1 * (self.step_count // 100))
        
        for name, idx in self.name_to_idx.items():
            if 'bn' in name.lower():
                continue
            
            current_lr = self.optimizer.param_groups[idx]['lr']
            target_lr = self.target_lrs.get(name, self.base_lr)
            new_lr = effective_momentum * current_lr + (1 - effective_momentum) * target_lr
            self.optimizer.param_groups[idx]['lr'] = new_lr
        
        if self.debug and self.step_count % 10 == 0 and boosted_names:
            print(f"  Boosted: {boosted_names[:4]}")
        
        self.bottleneck_history.append([
            f"{graph_builder.get_node_name(u)}->{graph_builder.get_node_name(v)}" 
            for u, v in cut_edges
        ])
    
    def apply_gradient_clipping(self, model, T_set: Set[int], graph_builder):
        pass
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_statistics(self) -> Dict:
        return {
            'bottleneck_history': self.bottleneck_history,
            'boost_counts': dict(self.boost_counts),
            'persistent_bottleneck': dict(self.persistent_bottleneck)
        }