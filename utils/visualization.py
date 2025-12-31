import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class PFNVisualizer:
    """Visualization tools for Parameter Flow Network analysis."""
    
    def __init__(self, save_dir: str = './results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_loss_vs_maxflow(self, losses: List[float], max_flows: List[float], 
                             title: str = "Loss vs Max Flow"):
        """Plot correlation between loss and max flow value."""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(losses, color='tab:red', label='Loss', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Max Flow', color='tab:blue')
        ax2.plot(max_flows, color='tab:blue', label='Max Flow', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        plt.title(title)
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_vs_maxflow.png'), dpi=150)
        plt.close()
    
    def plot_gradient_norms(self, gradient_norms: Dict[str, List[float]],
                           title: str = "Gradient Norms Over Time"):
        """Plot gradient norms for each layer."""
        plt.figure(figsize=(12, 6))
        
        for layer, norms in gradient_norms.items():
            if norms:
                plt.plot(norms, label=layer, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Gradient L2 Norm')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'gradient_norms.png'), dpi=150)
        plt.close()
    
    def plot_bottleneck_migration(self, bottleneck_history: List[List[str]],
                                  title: str = "Bottleneck Migration"):
        """Visualize how bottleneck position changes over training."""
        # Encode bottleneck positions
        all_bottlenecks = set()
        for b_list in bottleneck_history:
            all_bottlenecks.update(b_list)
        
        bottleneck_to_idx = {b: i for i, b in enumerate(sorted(all_bottlenecks))}
        
        # Create heatmap data
        heatmap = np.zeros((len(bottleneck_to_idx), len(bottleneck_history)))
        for t, b_list in enumerate(bottleneck_history):
            for b in b_list:
                heatmap[bottleneck_to_idx[b], t] = 1
        
        plt.figure(figsize=(14, 6))
        plt.imshow(heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.yticks(range(len(bottleneck_to_idx)), list(bottleneck_to_idx.keys()))
        plt.xlabel('PFN Analysis Step')
        plt.ylabel('Cut Edge')
        plt.title(title)
        plt.colorbar(label='In Min-Cut')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'bottleneck_migration.png'), dpi=150)
        plt.close()
    
    def plot_lr_dynamics(self, lr_history: Dict[str, List[float]],
                        title: str = "Learning Rate Dynamics"):
        """Plot learning rate changes per parameter group."""
        plt.figure(figsize=(12, 6))
        
        for name, lrs in lr_history.items():
            if lrs:
                plt.plot(lrs, label=name, alpha=0.7)
        
        plt.xlabel('PFN Analysis Step')
        plt.ylabel('Learning Rate')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'lr_dynamics.png'), dpi=150)
        plt.close()
    
    def plot_training_comparison(self, baseline_losses: List[float],
                                  pfn_losses: List[float],
                                  title: str = "Training Comparison"):
        """Compare baseline vs PFN training curves."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(baseline_losses, label='Baseline (Adam)', alpha=0.7)
        plt.plot(pfn_losses, label='PFN-Optimized', alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_comparison.png'), dpi=150)
        plt.close()
    
    def plot_network_graph(self, capacity: np.ndarray, cut_edges: List[Tuple[int, int]],
                          node_names: List[str], title: str = "PFN Graph"):
        """Simple visualization of the PFN graph structure."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Node positions (layered layout)
        positions = {
            0: (0, 0.5),      # Source
            1: (1, 0.8),      # L1_B0
            2: (1, 0.6),      # L1_B1
            3: (1, 0.4),      # L1_B2
            4: (1, 0.2),      # L1_B3
            5: (2, 0.65),     # L2_B0
            6: (2, 0.35),     # L2_B1
            7: (3, 0.5),      # L3_B0
            8: (4, 0.5),      # Sink
        }
        
        # Draw edges
        for i in range(capacity.shape[0]):
            for j in range(capacity.shape[1]):
                if capacity[i][j] > 1e-9:
                    x = [positions[i][0], positions[j][0]]
                    y = [positions[i][1], positions[j][1]]
                    
                    is_cut = (i, j) in cut_edges
                    color = 'red' if is_cut else 'gray'
                    width = 3 if is_cut else 1
                    
                    ax.plot(x, y, color=color, linewidth=width, alpha=0.6)
        
        # Draw nodes
        for node, (x, y) in positions.items():
            color = 'lightgreen' if node == 0 else ('lightcoral' if node == 8 else 'lightblue')
            circle = plt.Circle((x, y), 0.08, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node_names[node] if node < len(node_names) else str(node),
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'pfn_graph.png'), dpi=150)
        plt.close()
