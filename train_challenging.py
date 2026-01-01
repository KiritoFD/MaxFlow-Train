"""
Training script for challenging scenarios designed to show PFN advantages.
Supports MNIST, CIFAR-10, CIFAR-100.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
import numpy as np

from models.bottleneck_mlp import BottleneckMLP, DeepMLP
from models.partitioned_mlp import PartitionedMLP
from models.cnn_models import BottleneckCNN, DeepCNN, StandardCNN
from pfn import PFNGraphBuilder, DinicSolver, BottleneckOptimizer
from utils import (MetricsTracker, PFNVisualizer, get_device, get_device_info, 
                   get_memory_usage, get_dataloaders)


def get_model(model_type: str, dataset: str = 'mnist', **kwargs):
    """Factory function for models."""
    if dataset.lower() == 'mnist':
        if model_type == 'bottleneck':
            return BottleneckMLP(bottleneck_width=kwargs.get('bottleneck_width', 16))
        elif model_type == 'deep':
            return DeepMLP(num_hidden_layers=kwargs.get('num_layers', 10))
        else:
            return PartitionedMLP()
    else:  # CIFAR-10/100
        if model_type == 'bottleneck':
            return BottleneckCNN(bottleneck_channels=kwargs.get('bottleneck_width', 8))
        elif model_type == 'deep':
            return DeepCNN(num_layers=kwargs.get('num_layers', 10))
        else:
            return StandardCNN()


def train_epoch(model, train_loader, optimizer, criterion, device, 
                metrics, use_pfn=False, pfn_components=None, global_step=0,
                pfn_interval=20):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        if use_pfn:
            pfn_components['bottleneck_opt'].zero_grad()
        else:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # PFN Analysis
        if use_pfn and global_step % pfn_interval == 0 and global_step > 0:
            gradients = model.get_all_block_gradients()
            cpu_gradients = {
                layer: [g.detach().cpu() for g in grads]
                for layer, grads in gradients.items()
            }
            
            graph_builder = pfn_components['graph_builder']
            solver = pfn_components['solver']
            bottleneck_opt = pfn_components['bottleneck_opt']
            
            capacity, edge_info = graph_builder.build_graph(cpu_gradients)
            max_flow, cut_edges, S_set, T_set = solver.find_min_cut(
                capacity, graph_builder.source, graph_builder.sink
            )
            
            metrics.record_max_flow(max_flow)
            min_cut_cap = sum(capacity[u][v] for u, v in cut_edges)
            metrics.min_cut_capacity_history.append(min_cut_cap)
            
            bottleneck_opt.update_learning_rates(
                S_set, T_set, cut_edges, capacity, graph_builder
            )
            bottleneck_opt.apply_gradient_clipping(model, T_set, graph_builder)
        
        if use_pfn:
            pfn_components['bottleneck_opt'].step()
        else:
            optimizer.step()
        
        epoch_loss += loss.item()
        metrics.record_loss(loss.item())
        global_step += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(train_loader), global_step


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
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
    
    return correct / total


def run_challenging_experiment(
    dataset: str = 'cifar10',
    scenario: str = 'bottleneck',
    epochs: int = 20,
    batch_size: int = 128,
    pixel_noise: float = 0.0,
    label_noise: float = 0.0,
    samples_per_class: int = None,
    bottleneck_width: int = 8,
    num_layers: int = 10,
    force_cpu: bool = False,
    save_dir: str = './results/challenging',
    lr: float = 0.001
):
    """
    Run experiment with challenging scenarios.
    
    Args:
        dataset: 'mnist', 'cifar10', or 'cifar100'
        scenario: 'bottleneck', 'deep', or 'standard'
        epochs: Number of epochs
        batch_size: Batch size
        pixel_noise: Gaussian noise std (0.0-1.0)
        label_noise: Label flip ratio (0.0-1.0)
        samples_per_class: If set, limit training samples per class
        bottleneck_width: Width of bottleneck layer
        num_layers: Number of hidden layers (for 'deep' scenario)
        force_cpu: Force CPU usage
        save_dir: Save directory
        lr: Learning rate
    """
    os.makedirs(save_dir, exist_ok=True)
    device = get_device(force_cpu=force_cpu)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Device: {get_device_info(device)}")
    print(f"Scenario: {scenario}")
    print(f"Pixel noise: {pixel_noise}, Label noise: {label_noise}")
    if samples_per_class:
        print(f"Samples per class: {samples_per_class}")
    print(f"{'='*60}")
    
    # Data loading
    train_loader, test_loader, num_classes = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        pixel_noise=pixel_noise,
        label_noise=label_noise,
        samples_per_class=samples_per_class
    )
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    # ==================== BASELINE ====================
    print("\n" + "-" * 40)
    print("Training BASELINE")
    print("-" * 40)
    
    model_baseline = get_model(scenario, dataset, bottleneck_width=bottleneck_width, 
                               num_layers=num_layers).to(device)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler_baseline = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_baseline, T_max=epochs
    )
    
    metrics_baseline = MetricsTracker()
    
    global_step = 0
    for epoch in range(epochs):
        avg_loss, global_step = train_epoch(
            model_baseline, train_loader, optimizer_baseline, criterion,
            device, metrics_baseline, use_pfn=False, global_step=global_step
        )
        scheduler_baseline.step()
        accuracy = evaluate(model_baseline, test_loader, device)
        metrics_baseline.record_accuracy(accuracy)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    
    results['baseline'] = metrics_baseline.get_summary()
    
    # ==================== PFN ====================
    print("\n" + "-" * 40)
    print("Training with PFN")
    print("-" * 40)
    
    model_pfn = get_model(scenario, dataset, bottleneck_width=bottleneck_width,
                          num_layers=num_layers).to(device)
    param_groups = model_pfn.get_parameter_groups()
    
    # Update lr in param groups
    for g in param_groups:
        g['lr'] = lr
    
    base_optimizer = optim.Adam(param_groups, lr=lr)
    scheduler_pfn = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=epochs)
    
    graph_builder = PFNGraphBuilder(use_hessian_approx=True, depth_penalty=True)
    solver = DinicSolver()
    bottleneck_opt = BottleneckOptimizer(
        base_optimizer, base_lr=lr, boost_factor=2.0, 
        clip_value=1.0, enable_pruning=False
    )
    
    pfn_components = {
        'graph_builder': graph_builder,
        'solver': solver,
        'bottleneck_opt': bottleneck_opt
    }
    
    metrics_pfn = MetricsTracker()
    
    global_step = 0
    for epoch in range(epochs):
        avg_loss, global_step = train_epoch(
            model_pfn, train_loader, base_optimizer, criterion,
            device, metrics_pfn, use_pfn=True, pfn_components=pfn_components,
            global_step=global_step, pfn_interval=10
        )
        scheduler_pfn.step()
        accuracy = evaluate(model_pfn, test_loader, device)
        metrics_pfn.record_accuracy(accuracy)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    
    results['pfn'] = metrics_pfn.get_summary()
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    
    baseline_acc = results['baseline']['final_accuracy']
    pfn_acc = results['pfn']['final_accuracy']
    improvement = pfn_acc - baseline_acc
    
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
    print(f"PFN Accuracy:      {pfn_acc:.4f}")
    print(f"Improvement:       {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Save results
    results['config'] = {
        'dataset': dataset,
        'scenario': scenario,
        'epochs': epochs,
        'pixel_noise': pixel_noise,
        'label_noise': label_noise,
        'samples_per_class': samples_per_class,
        'bottleneck_width': bottleneck_width,
        'num_layers': num_layers,
        'lr': lr
    }
    results['improvement'] = improvement
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    visualizer = PFNVisualizer(save_dir=save_dir)
    visualizer.plot_training_comparison(
        metrics_baseline.loss_history,
        metrics_pfn.loss_history,
        title=f"{dataset.upper()} {scenario}: Baseline vs PFN"
    )
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Challenging PFN Experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--scenario', type=str, default='bottleneck',
                        choices=['bottleneck', 'deep', 'standard'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pixel_noise', type=float, default=0.0)
    parser.add_argument('--label_noise', type=float, default=0.0)
    parser.add_argument('--samples_per_class', type=int, default=None)
    parser.add_argument('--bottleneck_width', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    run_challenging_experiment(
        dataset=args.dataset,
        scenario=args.scenario,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pixel_noise=args.pixel_noise,
        label_noise=args.label_noise,
        samples_per_class=args.samples_per_class,
        bottleneck_width=args.bottleneck_width,
        num_layers=args.num_layers,
        force_cpu=args.cpu
    )
