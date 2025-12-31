"""
Complete PFN Experiment for MNIST

This script runs both baseline and PFN training, then generates comparison plots.
"""

import os
import json
import argparse

from train_baseline import train_baseline
from train_pfn import train_pfn
from utils import PFNVisualizer, get_device, get_device_info

def run_full_experiment(epochs: int = 10, batch_size: int = 512, force_cpu: bool = False):
    """Run complete experiment comparing baseline vs PFN."""
    
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Show device info
    device = get_device(force_cpu=force_cpu)
    print(f"Running experiment on: {get_device_info(device)}")
    print()
    
    print("=" * 60)
    print("Phase 1: Baseline Training (Standard Adam)")
    print("=" * 60)
    baseline_metrics = train_baseline(
        epochs=epochs, 
        batch_size=batch_size,
        save_dir=os.path.join(results_dir, 'baseline'),
        force_cpu=force_cpu
    )
    
    print("\n" + "=" * 60)
    print("Phase 2: PFN-Optimized Training")
    print("=" * 60)
    pfn_metrics = train_pfn(
        epochs=epochs,
        batch_size=batch_size,
        pfn_interval=20,
        save_dir=os.path.join(results_dir, 'pfn'),
        force_cpu=force_cpu
    )
    
    # Generate comparison plots
    print("\n" + "=" * 60)
    print("Phase 3: Generating Comparison Analysis")
    print("=" * 60)
    
    visualizer = PFNVisualizer(save_dir=results_dir)
    visualizer.plot_training_comparison(
        baseline_metrics.loss_history,
        pfn_metrics.loss_history,
        title="MNIST Training: Baseline vs PFN-Optimized"
    )
    
    # Final comparison summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    baseline_summary = baseline_metrics.get_summary()
    pfn_summary = pfn_metrics.get_summary()
    
    print(f"\nDevice: {get_device_info(device)}")
    print(f"\n{'Metric':<30} {'Baseline':>15} {'PFN':>15}")
    print("-" * 60)
    print(f"{'Final Accuracy':<30} {baseline_summary['final_accuracy']:>15.4f} "
          f"{pfn_summary['final_accuracy']:>15.4f}")
    print(f"{'Convergence Slope':<30} {baseline_summary['convergence_slope']:>15.6f} "
          f"{pfn_summary['convergence_slope']:>15.6f}")
    print(f"{'Final Loss':<30} {baseline_summary['final_loss']:>15.4f} "
          f"{pfn_summary['final_loss']:>15.4f}")
    
    if baseline_summary.get('peak_memory_mb'):
        print(f"{'Peak Memory (MB)':<30} {baseline_summary['peak_memory_mb']:>15.2f} "
              f"{pfn_summary['peak_memory_mb']:>15.2f}")
    
    combined_results = {
        'device': str(device),
        'baseline': baseline_summary,
        'pfn': pfn_summary,
        'improvement': {
            'accuracy_diff': pfn_summary['final_accuracy'] - baseline_summary['final_accuracy'],
            'convergence_improvement': pfn_summary['convergence_slope'] - baseline_summary['convergence_slope']
        }
    }
    
    with open(os.path.join(results_dir, 'comparison.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}/")
    print("Generated files:")
    print("  - baseline/metrics.json, model.pt")
    print("  - pfn/metrics.json, model.pt")
    print("  - pfn/loss_vs_maxflow.png")
    print("  - pfn/gradient_norms.png")
    print("  - pfn/bottleneck_migration.png")
    print("  - pfn/lr_dynamics.png")
    print("  - training_comparison.png")
    print("  - comparison.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PFN MNIST Experiment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    run_full_experiment(epochs=args.epochs, batch_size=args.batch_size, force_cpu=args.cpu)
