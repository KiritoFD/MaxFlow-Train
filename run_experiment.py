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
        'epochs_list': [30],
        'batches_list': [32],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 4},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.3, 'label_noise': 0.15, 'samples': 100},
        ]
    },
    'cifar10': {
        'epochs_list': [50],
        'batches_list': [64,128,256,512],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 2},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.2, 'label_noise': 0.1, 'samples': 200},
        ]
    },
    'cifar100': {
        'epochs_list': [100],
        'batches_list': [128,256,512],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 2},
            {'name': '2.Deep', 'scenario': 'deep', 'num_layers': 15, 'lr': 0.0005},
            {'name': '3.Noisy', 'scenario': 'standard', 'pixel_noise': 0.15, 'label_noise': 0.1, 'samples': 300},
        ]
    },
}
PFN_INTERVAL = 100  # 增加间隔，减少干扰Adam
PFN_WARMUP_STEPS = 200  # 前200步不介入，让Adam稳定
BASE_LR = 0.0001


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
        
        # PFN分析（warmup后才介入，降低频率）
        if pfn and step > PFN_WARMUP_STEPS and step % PFN_INTERVAL == 0:
            grads = {k: [t.detach().cpu() for t in v] for k, v in model.get_all_block_gradients().items()}
            
            pfn['gb'].current_epoch = epoch
            pfn['gb'].total_epochs = total_epochs
            
            cap, meta = pfn['gb'].build_graph(grads)
            max_flow, cuts, S, T = pfn['solver'].find_min_cut(cap, pfn['gb'].source, pfn['gb'].sink)
            
            # 计算flow deficit
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
            'gb': PFNGraphBuilder(history_size=5),  # 启用历史平滑
            'solver': IncrementalPushRelabel(),
            'opt': BottleneckOptimizer(opt, lr, base_boost=1.3, max_boost=3.0, decay_factor=0.95)
        }
        pfn['gb'].debug = (config['epochs'] <= 15)  # 短实验开启debug
        pfn['opt'].debug = (config['epochs'] <= 15)
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


def run_experiment(dataset='mnist', force_cpu=False, scenarios=None,
                   epochs_override=None, batch_override=None):
    device = get_device(force_cpu)
    cfg = CONFIG[dataset]
    # Determine effective epochs/batch for this single run (fallback to first elements)
    epochs = epochs_override if epochs_override is not None else cfg['epochs_list'][0]
    batch = batch_override if batch_override is not None else cfg['batches_list'][0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'./results_{timestamp}/{dataset}_e{epochs}_b{batch}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"  PFN EXPERIMENT - {dataset.upper()}")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch}")
    print("=" * 60)
    
    # 过滤要运行的场景
    scenario_configs = cfg['scenarios']
    if scenarios:
        scenario_configs = [s for s in scenario_configs if s['name'] in scenarios or any(kw in s['name'].lower() for kw in scenarios)]
    
    results = []
    for i, scenario_cfg in enumerate(scenario_configs):
        print(f"\n[{i+1}/{len(scenario_configs)}] {scenario_cfg['name']}")
        print("-" * 40)
        results.append(run_scenario(dataset, scenario_cfg, epochs, batch, device, results_dir))
        print(f"  >> Base={results[-1]['baseline']:.4f} | PFN={results[-1]['pfn']:.4f} | Δ={results[-1]['improvement']:+.4f}")
    
    # 汇总
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
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore cached baseline results')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (single value for all runs in this invocation)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Override batch size (single value for all runs in this invocation)')
    parser.add_argument('--epochs-list', nargs='+', type=int, default=None,
                        help='List of epochs to run all combinations, e.g., --epochs-list 10 20')
    parser.add_argument('--batches-list', nargs='+', type=int, default=None,
                        help='List of batch sizes to run all combinations, e.g., --batches-list 64 128')
    
    args = parser.parse_args()
    
    # 处理--no-cache选项
    if args.no_cache:
        baseline_cache.cache_dir = './results-baseline-nocache'
        os.makedirs(baseline_cache.cache_dir, exist_ok=True)
    
    # 确定目标数据集：优先 --datasets, 否则 --dataset，若都没给则默认全部数据集
    if args.datasets:
        targets = args.datasets
    elif args.dataset:
        targets = [args.dataset]
    else:
        targets = list(CONFIG.keys())
    
    # 对每个数据集，生成 epochs_list 与 batches_list（优先使用用户提供列表/单值，否则使用该数据集的配置）
    for ds in targets:
        cfg = CONFIG[ds]
        default_epochs = cfg['epochs_list']
        default_batches = cfg['batches_list']
        
        epochs_list = args.epochs_list if args.epochs_list is not None else ([args.epochs] if args.epochs is not None else default_epochs)
        batches_list = args.batches_list if args.batches_list is not None else ([args.batch] if args.batch is not None else default_batches)
        
        # 运行所有组合
        for ep in epochs_list:
            for bs in batches_list:
                print(f"\n>> Running dataset={ds} epochs={ep} batch={bs}")
                run_experiment(ds, args.cpu, args.scenarios, epochs_override=ep, batch_override=bs)
