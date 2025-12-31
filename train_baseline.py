import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from models.partitioned_mlp import PartitionedMLP
from utils.metrics import MetricsTracker
from utils.device import get_device, get_device_info, get_memory_usage

def train_baseline(epochs: int = 50, batch_size: int = 64, lr: float = 0.001,
                   save_dir: str = './results/baseline', force_cpu: bool = False):
    """Train MNIST with standard Adam optimizer (baseline)."""
    
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
    
    # Model and optimizer
    model = PartitionedMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    metrics = MetricsTracker()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Record gradient norms
            gradients = model.get_all_block_gradients()
            for layer, grads in gradients.items():
                for i, g in enumerate(grads):
                    metrics.record_gradient_norm(f"{layer}_block{i}", g.norm(2).item())
            
            optimizer.step()
            
            epoch_loss += loss.item()
            metrics.record_loss(loss.item())
            
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
    
    # Save results
    summary = metrics.get_summary()
    summary['convergence_slope'] = metrics.compute_convergence_slope()
    summary['device'] = str(device)
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'summary': summary,
            'loss_history': metrics.loss_history,
            'accuracy_history': metrics.accuracy_history
        }, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    
    print(f"\nBaseline Training Complete!")
    print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
    print(f"Convergence Slope: {summary['convergence_slope']:.6f}")
    
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Baseline MNIST Training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()
    
    train_baseline(epochs=args.epochs, batch_size=args.batch_size, force_cpu=args.cpu)
