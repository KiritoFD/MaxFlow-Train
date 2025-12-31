import numpy as np
from typing import Dict, List
from collections import defaultdict

class MetricsTracker:
    """Track and compute evaluation metrics for PFN experiments."""
    
    def __init__(self):
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.max_flow_history: List[float] = []
        self.min_cut_capacity_history: List[float] = []
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: List[float] = []
        
    def record_loss(self, loss: float):
        self.loss_history.append(loss)
    
    def record_accuracy(self, accuracy: float):
        self.accuracy_history.append(accuracy)
    
    def record_max_flow(self, max_flow: float):
        self.max_flow_history.append(max_flow)
    
    def record_gradient_norm(self, layer: str, norm: float):
        self.gradient_norms[layer].append(norm)
    
    def record_memory(self, memory_mb: float):
        self.memory_usage.append(memory_mb)
    
    def compute_convergence_slope(self, first_n_epochs: int = 10) -> float:
        """Compute average loss decrease rate in first N epochs."""
        if len(self.loss_history) < 2:
            return 0.0
        
        # Assuming ~steps_per_epoch steps per epoch
        n = min(len(self.loss_history), first_n_epochs * 100)
        if n < 2:
            return 0.0
        
        losses = np.array(self.loss_history[:n])
        x = np.arange(n)
        slope = np.polyfit(x, losses, 1)[0]
        return -slope  # Negative slope means decreasing loss
    
    def compute_topological_stability(self, bottleneck_history: List[List[str]]) -> float:
        """
        Compute stability of min-cut position.
        Lower value = more stable (less fluctuation).
        """
        if len(bottleneck_history) < 2:
            return 0.0
        
        changes = 0
        for i in range(1, len(bottleneck_history)):
            prev = set(bottleneck_history[i-1])
            curr = set(bottleneck_history[i])
            if prev != curr:
                changes += 1
        
        stability = 1.0 - (changes / (len(bottleneck_history) - 1))
        return stability
    
    def compute_dual_gap(self) -> List[float]:
        """
        Compute dual gap between max flow and min cut capacity.
        Should be ~0 by max-flow min-cut theorem.
        """
        gaps = []
        for mf, mc in zip(self.max_flow_history, self.min_cut_capacity_history):
            gaps.append(abs(mf - mc))
        return gaps
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'convergence_slope': self.compute_convergence_slope(),
            'avg_max_flow': np.mean(self.max_flow_history) if self.max_flow_history else None,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else None
        }
