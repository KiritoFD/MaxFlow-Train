"""PFN Experiment - Parameter Flow Network for Neural Network Optimization."""

import os, json, argparse, torch, numpy as np
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional
from models import get_model
from pfn import PFNGraphBuilder, IncrementalPushRelabel, BottleneckOptimizer


# ==================== CONFIG ====================
CONFIG = {
    'mnist': {
        'fast': {'epochs': 10, 'batch': 128},
        'full': {'epochs': 20, 'batch': 64},
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 8},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.3, 'label_noise': 0.15, 'samples': 100},
        ]
    },
    'cifar10': {
        'fast': {'epochs': 10, 'batch': 128},
        'full': {'epochs': 20, 'batch': 128},
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 4},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},  # 加深到15层
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.2, 'label_noise': 0.1, 'samples': 200},
        ]
    },
    'cifar100': {
        'fast': {'epochs': 15, 'batch': 128},
        'full': {'epochs': 40, 'batch': 128},
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 4},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},  # 加深到15层
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.15, 'label_noise': 0.1, 'samples': 300},
        ]
    },
}
PFN_INTERVAL = 50  # 每50步介入一次，给Adam喘息时间
BASE_LR = 0.001


# ==================== BASELINE CACHE ====================
class BaselineCache:
    """Baseline结果缓存系统"""
    
    def __init__(self, cache_dir: str = './results-baseline'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, dataset: str, scenario: str, batch_size: int, epochs: int) -> str:
        """生成缓存键"""
        return f"{dataset}_{scenario}_b{batch_size}_e{epochs}"
    
    def get(self, dataset: str, scenario: str, batch_size: int, epochs: int) -> Optional[Dict]:
        """获取缓存的baseline结果"""
        key = self._get_cache_key(dataset, scenario, batch_size, epochs)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def save(self, dataset: str, scenario: str, batch_size: int, epochs: int, 
             acc: float, loss_hist: List[float], acc_hist: List[float]):
        """保存baseline结果"""
        key = self._get_cache_key(dataset, scenario, batch_size, epochs)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        data = {
            'acc': float(acc),
            'loss': loss_hist,
            'acc_hist': acc_hist,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)


baseline_cache = BaselineCache()


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
    num_classes = 100 if dataset_name == 'cifar100' else 10
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, transform=transform)
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        DS = datasets.CIFAR100 if dataset_name == 'cifar100' else datasets.CIFAR10
        train_ds = DS('./data', train=True, download=True, transform=train_transform)
        test_ds = DS('./data', train=False, transform=test_transform)
    
    # Apply modifications in order: small sample first, then noise
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
        
        # PFN分析（降低频率）
        if pfn and step % PFN_INTERVAL == 0 and step > 0:
            grads = {k: [t.detach().cpu() for t in v] for k, v in model.get_all_block_gradients().items()}
            
            pfn['gb'].current_epoch = epoch
            pfn['gb'].total_epochs = total_epochs
            
            cap, _ = pfn['gb'].build_graph(grads)
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


def train_model(config: dict, device, use_pfn=False):
    lr = config.get('lr', BASE_LR)
    
    train_loader, test_loader = get_loaders(
        config['dataset'], config['batch_size'],
        config.get('pixel_noise', 0), config.get('label_noise', 0), config.get('samples')
    )
    
    model = get_model(
        config['scenario'], config['dataset'],
        bottleneck_width=config.get('bottleneck_width', 8),
        num_layers=config.get('num_layers', 10)
    ).to(device)
    
    if use_pfn:
        groups = model.get_parameter_groups()
        for g in groups: g['lr'] = lr
        opt = optim.Adam(groups, lr=lr)
        pfn = {
            'gb': PFNGraphBuilder(),
            'solver': IncrementalPushRelabel(),
            'opt': BottleneckOptimizer(opt, lr, base_boost=1.5)  # 温和boost
        }
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
        pfn = None
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'])
    step = 0
    loss_hist, acc_hist = [], []
    
    for ep in range(config['epochs']):
        loss, step = train_epoch(model, train_loader, opt, nn.CrossEntropyLoss(), device, pfn, step, ep, config['epochs'])
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        loss_hist.append(loss)
        acc_hist.append(acc)
        print(f"    [{ep+1:2d}/{config['epochs']}] loss={loss:.4f} acc={acc:.4f}")
    
    return acc, loss_hist, acc_hist


def run_scenario(dataset, cfg, epochs, batch, device, results_dir):
    name = cfg['name']
    os.makedirs(os.path.join(results_dir, name), exist_ok=True)
    
    config = {
        'dataset': dataset, 'scenario': cfg['scenario'],
        'epochs': cfg.get('epochs', epochs), 'batch_size': cfg.get('batch_size', batch),
        'lr': cfg.get('lr', BASE_LR), 'bottleneck_width': cfg.get('bottleneck_width', 8),
        'num_layers': cfg.get('num_layers', 10), 'pixel_noise': cfg.get('pixel_noise', 0),
        'label_noise': cfg.get('label_noise', 0), 'samples': cfg.get('samples')
    }
    
    # 检查baseline缓存
    print(f"  [Baseline]", end=' ')
    cached = baseline_cache.get(dataset, cfg['scenario'], config['batch_size'], config['epochs'])
    
    if cached:
        print("(缓存)")
        base_acc = cached['acc']
        base_loss = cached['loss']
        base_acc_hist = cached['acc_hist']
    else:
        print("(训练)")
        base_acc, base_loss, base_acc_hist = train_model(config, device, False)
        # 保存到缓存
        baseline_cache.save(dataset, cfg['scenario'], config['batch_size'], config['epochs'],
                           base_acc, base_loss, base_acc_hist)
    
    print(f"  [PFN]")
    pfn_acc, pfn_loss, pfn_acc_hist = train_model(config, device, True)
    
    # 保存
    with open(os.path.join(results_dir, name, 'log.json'), 'w') as f:
        json.dump({'baseline': {'loss': base_loss, 'acc': base_acc_hist}, 'pfn': {'loss': pfn_loss, 'acc': pfn_acc_hist}}, f)
    
    # 画图
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


def run_experiment(dataset='mnist', force_cpu=False, fast=False):
    device = get_device(force_cpu)
    mode = 'FAST' if fast else 'FULL'
    cfg = CONFIG[dataset]
    settings = cfg['fast' if fast else 'full']
    epochs, batch = settings['epochs'], settings['batch']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'./results_{timestamp}/{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"  PFN EXPERIMENT - {dataset.upper()} - {mode}")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch}")
    print("=" * 60)
    
    results = []
    for i, scenario_cfg in enumerate(cfg['scenarios']):
        print(f"\n[{i+1}/{len(cfg['scenarios'])}] {scenario_cfg['name']}")
        print("-" * 40)
        results.append(run_scenario(dataset, scenario_cfg, epochs, batch, device, results_dir))
        print(f"  >> Base={results[-1]['baseline']:.4f} | PFN={results[-1]['pfn']:.4f} | Δ={results[-1]['improvement']:+.4f}")
    
    # 汇总
    print("\n" + "=" * 60)
    wins = sum(1 for r in results if r['improvement'] > 0)
    avg = sum(r['improvement'] for r in results) / len(results)
    for r in results:
        print(f"{r['name']:<20} {r['baseline']:.4f}  {r['pfn']:.4f}  {r['improvement']:+.4f} {'✓' if r['improvement'] > 0 else '✗'}")
    print("-" * 60)
    print(f"AVG: {avg:+.4f}  ({wins}/{len(results)} wins)")
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump({'results': results, 'avg': avg, 'wins': wins}, f, indent=2)
    
    print(f"\nSaved: {results_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    run_experiment(args.dataset, args.cpu, args.fast)
