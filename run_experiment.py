#%%writefile run_experiment.py
"""PFN Experiment - Parameter Flow Network for Neural Network Optimization."""

import os, json, argparse, torch, numpy as np
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional
from models import get_model
from pfn import PFNGraphBuilder, IncrementalPushRelabel, BottleneckOptimizer, reset_pfn_logger, get_pfn_logger


# ==================== CONFIG ====================
CONFIG = {
    'mnist': {
        'epochs_list': [30],
        'batches_list': [64],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 4},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.3, 'label_noise': 0.15, 'samples': 100},
        ]
    },
    'cifar10': {
        'epochs_list': [50],
        'batches_list': [256],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.NoBN', 'scenario': 'standard_no_bn'},
            {'name': '2.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 2},
            {'name': '3.Deep', 'scenario': 'deep', 'num_layers': 12, 'lr': 0.0003},
            {'name': '4.Noisy', 'scenario': 'standard', 'pixel_noise': 0.2, 'label_noise': 0.1, 'samples': 200},
        ]
    },
    'cifar100': {
        'epochs_list': [200],
        'batches_list': [128, 256, 512],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.NoBN', 'scenario': 'standard_no_bn'},
            {'name': '2.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 2},
            {'name': '3.Deep', 'scenario': 'deep', 'num_layers': 12, 'lr': 0.0003},
            {'name': '4.Noisy', 'scenario': 'standard', 'pixel_noise': 0.15, 'label_noise': 0.1, 'samples': 300},
        ]
    },
}
PFN_INTERVAL = 50
PFN_WARMUP_STEPS = 100
BASE_LR = 0.0001


# ==================== DATA ====================
class NoisyDataset(Dataset):
    """Dataset wrapper with pixel and label noise."""
    def __init__(self, base_dataset, pixel_noise=0.0, label_noise=0.0, num_classes=10):
        self.base = base_dataset
        self.pixel_noise = pixel_noise
        self.num_classes = num_classes
        
        self.noisy_labels = []
        np.random.seed(42)
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label_noise > 0 and np.random.random() < label_noise:
                candidates = [l for l in range(num_classes) if l != label]
                label = np.random.choice(candidates)
            self.noisy_labels.append(label)
    
    def __len__(self): return len(self.base)
    
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        label = self.noisy_labels[idx]
        if self.pixel_noise > 0:
            img = img + torch.randn_like(img) * self.pixel_noise
        return img, label


class SmallDataset(Dataset):
    """Dataset wrapper that limits samples per class."""
    def __init__(self, base_dataset, samples_per_class, num_classes=10):
        self.base = base_dataset
        indices_per_class = {c: [] for c in range(num_classes)}
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            label = label.item() if isinstance(label, torch.Tensor) else label
            if len(indices_per_class[label]) < samples_per_class:
                indices_per_class[label].append(idx)
        self.indices = [i for c in range(num_classes) for i in indices_per_class[c]]
    
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.base[self.indices[idx]]


def get_loaders(dataset_name, batch_size, pixel_noise=0, label_noise=0, samples_per_class=None):
    """适配多环境的数据加载器，支持 Kaggle/本地/下载回退"""
    num_classes = 100 if dataset_name == 'cifar100' else 10
    
    writable_root = '/kaggle/working/data'
    try:
        os.makedirs(writable_root, exist_ok=True)
        print(f"Using writable root: {writable_root}")
    except Exception as e:
        fallback_writable = os.path.join(os.getcwd(), 'data_writable')
        os.makedirs(fallback_writable, exist_ok=True)
        print(f"Warning: cannot create {writable_root} ({e}); falling back to {fallback_writable}")
        writable_root = fallback_writable
    
    local_root = os.path.join(os.getcwd(), 'data')
    input_root = f'/kaggle/input/{dataset_name}'
    candidates = [input_root, local_root, writable_root]
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        def try_load_mnist(train):
            for root in candidates:
                if not os.path.exists(root): continue
                try:
                    ds = datasets.MNIST(root, train=train, download=False, transform=transform)
                    print(f"    Loaded MNIST (train={train}) from {root}")
                    return ds
                except Exception as e:
                    print(f"    MNIST load failed at {root}: {e}")
            ds = datasets.MNIST(writable_root, train=train, download=True, transform=transform)
            print(f"    Downloaded MNIST to {writable_root}")
            return ds
        
        train_ds = try_load_mnist(True)
        test_ds = try_load_mnist(False)
        
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        DS = datasets.CIFAR100 if dataset_name == 'cifar100' else datasets.CIFAR10
        
        def try_load_cifar(train):
            for root in candidates:
                if not os.path.exists(root): continue
                try:
                    ds = DS(root=root, train=train, download=True, 
                           transform=(train_transform if train else test_transform))
                    print(f"    Loaded {dataset_name.upper()} (train={train}) from {root}")
                    return ds
                except Exception as e:
                    print(f"    {dataset_name.upper()} load failed at {root}: {e}")
            ds = DS(root=writable_root, train=train, download=True,
                   transform=(train_transform if train else test_transform))
            print(f"    Downloaded {dataset_name.upper()} to {writable_root}")
            return ds
        
        train_ds = try_load_cifar(True)
        test_ds = try_load_cifar(False)
    
    if samples_per_class:
        train_ds = SmallDataset(train_ds, samples_per_class, num_classes)
    if pixel_noise > 0 or label_noise > 0:
        train_ds = NoisyDataset(train_ds, pixel_noise, label_noise, num_classes)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size * 2, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ==================== DEVICE ====================
def get_device(force_cpu=False):
    if force_cpu: return torch.device('cpu')
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


# ==================== TRAINING ====================
def train_epoch(model, loader, opt, criterion, device, pfn=None, step=0, epoch=0, total_epochs=1):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="    train", leave=False):
        x, y = x.to(device), y.to(device)
        
        (pfn['opt'] if pfn else opt).zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        
        if pfn and step > PFN_WARMUP_STEPS and step % PFN_INTERVAL == 0:
            grads = {k: [t.detach().cpu() for t in v] for k, v in model.get_all_block_gradients().items()}
            
            pfn['gb'].current_epoch = epoch
            pfn['gb'].total_epochs = total_epochs
            
            cap, meta = pfn['gb'].build_graph(grads)
            max_flow, cuts, S, T = pfn['solver'].find_min_cut(cap, pfn['gb'].source, pfn['gb'].sink)
            
            total_cap = sum(cap[u][v] for u, v in cuts) if cuts else 1.0
            flow_deficit = (total_cap - max_flow) / (total_cap + 1e-9)
            
            pfn['opt'].update_learning_rates(S, T, cuts, cap, pfn['gb'], flow_deficit)
        
        (pfn['opt'] if pfn else opt).step()
        total_loss += loss.item()
        step += 1
    
    return total_loss / len(loader), step


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(config: dict, device, use_pfn=False, results_dir: str = './results'):
    lr = config.get('lr', BASE_LR)
    
    train_loader, test_loader = get_loaders(
        config['dataset'], config['batch_size'],
        config.get('pixel_noise', 0), config.get('label_noise', 0), config.get('samples')
    )
    
    model = get_model(
        config['scenario'], config['dataset'],
        bottleneck_width=config.get('bottleneck_width', 8),
        num_layers=config.get('num_layers', 10),
        auto_partition=True,
        target_width=16
    ).to(device)
    
    pfn_logger = None
    if use_pfn:
        pfn_log_dir = os.path.join(results_dir, 'pfn_logs')
        pfn_logger = reset_pfn_logger(pfn_log_dir, enabled=True)
        
        groups = model.get_parameter_groups()
        for g in groups: g['lr'] = lr
        opt = optim.Adam(groups, lr=lr)
        
        gb = PFNGraphBuilder(history_size=5)
        gb.set_logger(pfn_logger)
        
        bn_opt = BottleneckOptimizer(
            opt, lr, 
            base_boost=1.5,
            max_boost=4.0, 
            decay_factor=0.92,
            lr_momentum=0.8
        )
        bn_opt.set_logger(pfn_logger)
        
        pfn = {
            'gb': gb,
            'solver': IncrementalPushRelabel(),
            'opt': bn_opt,
            'logger': pfn_logger
        }
        pfn['gb'].debug = (config['epochs'] <= 20)
        pfn['opt'].debug = (config['epochs'] <= 20)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
        pfn = None
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'])
    step = 0
    loss_hist, acc_hist = [], []
    
    for ep in range(config['epochs']):
        if pfn:
            pfn['opt'].current_epoch = ep
        
        loss, step = train_epoch(model, train_loader, opt, nn.CrossEntropyLoss(), device, pfn, step, ep, config['epochs'])
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        loss_hist.append(loss)
        acc_hist.append(acc)
        print(f"    [{ep+1:2d}/{config['epochs']}] loss={loss:.4f} acc={acc:.4f}")
        
        if pfn_logger:
            pfn_logger.log_epoch_summary(ep, loss, acc)
    
    if pfn_logger:
        pfn_logger.save_summary()
    
    return acc, loss_hist, acc_hist


def run_scenario(dataset, cfg, epochs, batch, device, results_dir):
    name = cfg['name']
    scenario_dir = os.path.join(results_dir, name)
    os.makedirs(scenario_dir, exist_ok=True)
    
    config = {
        'dataset': dataset, 'scenario': cfg['scenario'],
        'epochs': cfg.get('epochs', epochs), 'batch_size': cfg.get('batch_size', batch),
        'lr': cfg.get('lr', BASE_LR), 'bottleneck_width': cfg.get('bottleneck_width', 8),
        'num_layers': cfg.get('num_layers', 10), 'pixel_noise': cfg.get('pixel_noise', 0),
        'label_noise': cfg.get('label_noise', 0), 'samples': cfg.get('samples')
    }
    
    print(f"  [Baseline]")
    base_acc, base_loss, base_acc_hist = train_model(config, device, False, scenario_dir)
    
    print(f"  [PFN]")
    pfn_acc, pfn_loss, pfn_acc_hist = train_model(config, device, True, scenario_dir)
    
    with open(os.path.join(results_dir, name, 'log.json'), 'w') as f:
        json.dump({'baseline': {'loss': base_loss, 'acc': base_acc_hist}, 'pfn': {'loss': pfn_loss, 'acc': pfn_acc_hist}}, f)
    
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        eps = range(1, config['epochs'] + 1)
        ax1.plot(eps, base_loss, 'o-', label='Baseline'); ax1.plot(eps, pfn_loss, 's-', label='PFN')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(eps, base_acc_hist, 'o-', label='Baseline'); ax2.plot(eps, pfn_acc_hist, 's-', label='PFN')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, name, 'curves.png'), dpi=150); plt.close()
    except: pass
    
    return {'name': name, 'baseline': base_acc, 'pfn': pfn_acc, 'improvement': pfn_acc - base_acc}


def run_experiment(dataset='mnist', force_cpu=False, scenarios=None,
                   epochs_override=None, batch_override=None):
    device = get_device(force_cpu)
    cfg = CONFIG[dataset]
    epochs = epochs_override if epochs_override is not None else cfg['epochs_list'][0]
    batch = batch_override if batch_override is not None else cfg['batches_list'][0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'./results_{timestamp}/{dataset}_e{epochs}_b{batch}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"  PFN EXPERIMENT - {dataset.upper()}")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch}")
    print("=" * 60)
    
    scenario_configs = cfg['scenarios']
    if scenarios:
        scenario_configs = [s for s in scenario_configs if s['name'] in scenarios or any(kw in s['name'].lower() for kw in scenarios)]
    
    results = []
    for i, scenario_cfg in enumerate(scenario_configs):
        print(f"\n[{i+1}/{len(scenario_configs)}] {scenario_cfg['name']}")
        print("-" * 40)
        results.append(run_scenario(dataset, scenario_cfg, epochs, batch, device, results_dir))
        print(f"  >> Base={results[-1]['baseline']:.4f} | PFN={results[-1]['pfn']:.4f} | Δ={results[-1]['improvement']:+.4f}")
    
    print("\n" + "=" * 60)
    wins = sum(1 for r in results if r['improvement'] > 0)
    avg = sum(r['improvement'] for r in results) / len(results) if results else 0
    for r in results:
        print(f"{r['name']:<20} {r['baseline']:.4f}  {r['pfn']:.4f}  {r['improvement']:+.4f} {'✓' if r['improvement'] > 0 else '✗'}")
    print("-" * 60)
    print(f"AVG: {avg:+.4f}  ({wins}/{len(results)} wins)")
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump({'results': results, 'avg': avg, 'wins': wins}, f, indent=2)
    
    print(f"\nSaved: {results_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PFN Experiment Runner')
    parser.add_argument('--dataset', default=None, choices=['mnist', 'cifar10', 'cifar100'],
                        help='Dataset to use (default: all datasets if omitted)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Run multiple datasets, e.g., --datasets mnist cifar10 (overrides --dataset)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='Specific scenarios to run (e.g., --scenarios 0 1 or --scenarios original bottleneck)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (single value for all runs in this invocation)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Override batch size (single value for all runs in this invocation)')
    parser.add_argument('--epochs-list', nargs='+', type=int, default=None,
                        help='List of epochs to run all combinations, e.g., --epochs-list 10 20')
    parser.add_argument('--batches-list', nargs='+', type=int, default=None,
                        help='List of batch sizes to run all combinations, e.g., --batches-list 64 128')
    
    args = parser.parse_args()
    
    if args.datasets:
        targets = args.datasets
    elif args.dataset:
        targets = [args.dataset]
    else:
        targets = list(CONFIG.keys())
    
    for ds in targets:
        cfg = CONFIG[ds]
        default_epochs = cfg['epochs_list']
        default_batches = cfg['batches_list']
        
        epochs_list = args.epochs_list if args.epochs_list is not None else ([args.epochs] if args.epochs is not None else default_epochs)
        batches_list = args.batches_list if args.batches_list is not None else ([args.batch] if args.batch is not None else default_batches)
        
        for ep in epochs_list:
            for bs in batches_list:
                print(f"\n>> Running dataset={ds} epochs={ep} batch={bs}")
                run_experiment(ds, args.cpu, args.scenarios, epochs_override=ep, batch_override=bs)