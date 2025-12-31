import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import numpy as np

from models.partitioned_mlp import PartitionedMLP
from pfn import PFNGraphBuilder, DinicSolver, BottleneckOptimizer
from utils import MetricsTracker, PFNVisualizer, get_device, get_device_info, get_memory_usage

def train_pfn(epochs: int = 50, batch_size: int = 64, lr: float = 0.001,
              pfn_interval: int = 20, save_dir: str = './results/pfn',
              force_cpu: bool = False):
    """
    Train MNIST with PFN-based bottleneck optimization.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Base learning rate
        pfn_interval: Run PFN analysis every N iterations
        save_dir: Directory to save results
        force_cpu: If True, force CPU usage
    """
    
    os.makedirs(save_dir, exist_ok=True)
    device = get_device(force_cpu=force_cpu)
    print(f"Using device: {get_device_info(device)}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model with parameter groups
    model = PartitionedMLP().to(device)
    param_groups = model.get_parameter_groups()
    
    # Base optimizer
    base_optimizer = optim.Adam(param_groups, lr=lr)
    
    # PFN components
    graph_builder = PFNGraphBuilder(gradient_history_size=10)
    solver = DinicSolver()
    bottleneck_opt = BottleneckOptimizer(base_optimizer, base_lr=lr,
                                         boost_factor=1.5, clip_value=1.0)
    
    # Metrics and visualization
    metrics = MetricsTracker()
    visualizer = PFNVisualizer(save_dir=save_dir)
    
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            bottleneck_opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Get gradients for PFN analysis
            gradients = model.get_all_block_gradients()
            
            # Record gradient norms
            for layer, grads in gradients.items():
                for i, g in enumerate(grads):
                    metrics.record_gradient_norm(f"{layer}_block{i}", g.norm(2).item())
            
            # PFN Analysis (every pfn_interval steps)
            if global_step % pfn_interval == 0 and global_step > 0:
                # Move gradients to CPU for graph analysis (works on any device)
                cpu_gradients = {
                    layer: [g.detach().cpu() for g in grads]
                    for layer, grads in gradients.items()
                }
                
                # Build PFN graph
                capacity, edge_info = graph_builder.build_graph(cpu_gradients)
                
                # Solve min-cut
                max_flow, cut_edges, S_set, T_set = solver.find_min_cut(
                    capacity, graph_builder.source, graph_builder.sink
                )
                
                # Record metrics
                metrics.record_max_flow(max_flow)
                min_cut_cap = sum(capacity[u][v] for u, v in cut_edges)
                metrics.min_cut_capacity_history.append(min_cut_cap)
                
                # Update learning rates based on bottleneck
                bottleneck_opt.update_learning_rates(S_set, T_set, cut_edges)
                
                # Apply gradient clipping to T_set
                bottleneck_opt.apply_gradient_clipping(model, T_set)
                
                # Log bottleneck info
                bottleneck_names = solver.get_bottleneck_layers(cut_edges, graph_builder)
                if batch_idx % (pfn_interval * 5) == 0:
                    print(f"\n  [PFN] Max Flow: {max_flow:.4f}, "
                          f"Bottleneck: {bottleneck_names}")
            
            bottleneck_opt.step()
            
            epoch_loss += loss.item()
            metrics.record_loss(loss.item())
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        metrics.record_accuracy(accuracy)
        
        # Memory tracking
        mem_usage = get_memory_usage(device)
        if mem_usage is not None:
            metrics.record_memory(mem_usage)
        
        print(f"Epoch {epoch+1}: Avg Loss = {epoch_loss/len(train_loader):.4f}, "
              f"Test Accuracy = {accuracy:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    opt_stats = bottleneck_opt.get_statistics()
    
    if metrics.max_flow_history:
        # Align loss and max_flow for plotting
        sampled_losses = metrics.loss_history[::pfn_interval][1:len(metrics.max_flow_history)+1]
        if len(sampled_losses) == len(metrics.max_flow_history):
            visualizer.plot_loss_vs_maxflow(sampled_losses, metrics.max_flow_history)
    
    visualizer.plot_gradient_norms(dict(metrics.gradient_norms))
    
    if opt_stats['bottleneck_history']:
        visualizer.plot_bottleneck_migration(opt_stats['bottleneck_history'])
    
    if opt_stats['lr_history']:
        visualizer.plot_lr_dynamics(opt_stats['lr_history'])
    
    # Save results
    summary = metrics.get_summary()
    summary['convergence_slope'] = metrics.compute_convergence_slope()
    summary['device'] = str(device)
    
    if opt_stats['bottleneck_history']:
        summary['topological_stability'] = metrics.compute_topological_stability(
            opt_stats['bottleneck_history']
        )
        summary['bottleneck_frequency'] = opt_stats['bottleneck_frequency']
    
    dual_gaps = metrics.compute_dual_gap()
    if dual_gaps:
        summary['avg_dual_gap'] = float(np.mean(dual_gaps))
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'summary': summary,
            'loss_history': metrics.loss_history,
            'accuracy_history': metrics.accuracy_history,
            'max_flow_history': metrics.max_flow_history
        }, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    
    print(f"\nPFN Training Complete!")
    print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
    print(f"Convergence Slope: {summary['convergence_slope']:.6f}")
    if 'topological_stability' in summary:
        print(f"Topological Stability: {summary['topological_stability']:.4f}")
    if 'avg_dual_gap' in summary:
        print(f"Avg Dual Gap: {summary['avg_dual_gap']:.6f}")
    
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PFN MNIST Training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()
    
    train_pfn(epochs=args.epochs, batch_size=args.batch_size, force_cpu=args.cpu)
