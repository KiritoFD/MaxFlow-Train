"""PFN Logger - 记录最小割和流网络优化过程"""

import os
import json
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import numpy as np


class PFNLogger:
    """PFN优化过程日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enabled = True
        
        # 日志数据结构
        self.cut_history = []  # 最小割历史
        self.flow_history = []  # 流量历史
        self.lr_history = []   # 学习率历史
        self.bottleneck_history = []  # 瓶颈节点历史
        self.epoch_summary = []  # 每个epoch的总结
    
    def log_min_cut(self, step: int, epoch: int, max_flow: float, cut_edges: List[Tuple[int, int]],
                    S_set: Set[int], T_set: Set[int], capacity_matrix: np.ndarray,
                    graph_builder, node_names_map: Dict[int, str]):
        """记录最小割信息"""
        if not self.enabled:
            return
        
        cut_info = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'max_flow': float(max_flow),
            'total_capacity': float(np.sum(capacity_matrix[graph_builder.source, :])),
            'S_size': len(S_set),
            'T_size': len(T_set),
            'num_cut_edges': len(cut_edges),
            'cut_edges': [
                {
                    'from': node_names_map.get(u, f'node_{u}'),
                    'to': node_names_map.get(v, f'node_{v}'),
                    'capacity': float(capacity_matrix[u][v])
                }
                for u, v in cut_edges
            ],
            'S_nodes': sorted([node_names_map.get(n, f'node_{n}') for n in S_set]),
            'T_nodes': sorted([node_names_map.get(n, f'node_{n}') for n in T_set])
        }
        
        self.cut_history.append(cut_info)
    
    def log_learning_rates(self, step: int, epoch: int, optimizer_groups: List[Dict]):
        """记录学习率变化"""
        if not self.enabled:
            return
        
        lr_info = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'learning_rates': {}
        }
        
        for i, group in enumerate(optimizer_groups):
            name = group.get('name', f'param_group_{i}')
            lr_info['learning_rates'][name] = float(group.get('lr', 0))
        
        self.lr_history.append(lr_info)
    
    def log_bottleneck_nodes(self, step: int, epoch: int, bottleneck_nodes: Dict[str, float]):
        """记录瓶颈节点及其boost倍数"""
        if not self.enabled:
            return
        
        bottleneck_info = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'bottleneck_nodes': bottleneck_nodes
        }
        
        self.bottleneck_history.append(bottleneck_info)
    
    def log_epoch_summary(self, epoch: int, loss: float, acc: float, additional_info: Optional[Dict] = None):
        """记录每个epoch的总结"""
        if not self.enabled:
            return
        
        summary = {
            'epoch': epoch,
            'loss': float(loss),
            'accuracy': float(acc),
            'timestamp': datetime.now().isoformat(),
        }
        
        if additional_info:
            summary.update(additional_info)
        
        self.epoch_summary.append(summary)
    
    def save_summary(self):
        """保存所有日志到文件"""
        if not self.enabled:
            return
        
        summary_file = os.path.join(self.log_dir, f'pfn_log_{self.timestamp}.json')
        
        output = {
            'timestamp': self.timestamp,
            'cut_history': self.cut_history,
            'flow_history': self.flow_history,
            'lr_history': self.lr_history,
            'bottleneck_history': self.bottleneck_history,
            'epoch_summary': self.epoch_summary
        }
        
        with open(summary_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[PFN] Saved log to {summary_file}")
    
    def log_raw_cut_details(self, step: int, epoch: int, cut_edges: List[Tuple[int, int]],
                            capacity_matrix: np.ndarray):
        """记录原始最小割细节"""
        if not self.enabled:
            return
        
        details_file = os.path.join(self.log_dir, f'cut_details_step{step:05d}.json')
        
        cut_details = {
            'step': step,
            'epoch': epoch,
            'num_edges': len(cut_edges),
            'edge_capacities': [
                {'from': int(u), 'to': int(v), 'capacity': float(capacity_matrix[u][v])}
                for u, v in cut_edges
            ]
        }
        
        with open(details_file, 'w') as f:
            json.dump(cut_details, f, indent=2)


def reset_pfn_logger(log_dir: str, enabled: bool = True) -> PFNLogger:
    """创建并初始化PFN日志记录器"""
    logger = PFNLogger(log_dir)
    logger.enabled = enabled
    return logger


def get_pfn_logger(log_dir: str = './pfn_logs') -> PFNLogger:
    """获取或创建PFN日志记录器"""
    return reset_pfn_logger(log_dir, enabled=True)
