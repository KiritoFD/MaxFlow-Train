#%%writefile run_experiment.py
"""PFN Experiment - Parameter Flow Network for Neural Network Optimization."""

import os, json, argparse, torch, numpy as np
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from models import get_model
from pfn import PFNGraphBuilder, IncrementalPushRelabel, BottleneckOptimizer, PFNLogger


# ==================== CONFIG ====================
CONFIG = {
    'cifar10': {
        'epochs_list': [150],
        'batches_list': [256,512],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.NoBN', 'scenario': 'standard_no_bn'},
            {'name': '2.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 2},
            {'name': '3.Deep', 'scenario': 'deep', 'num_layers': 12, 'lr': 0.0003},
            {'name': '4.Noisy', 'scenario': 'noisy', 'pixel_noise': 0.2, 'label_noise': 0.1, 'samples': 200},
        ]
    },
    'cifar100': {
        'epochs_list': [250],
        'batches_list': [1024],
        'scenarios': [
            {'name': '0.Original', 'scenario': 'standard'},
            {'name': '1.NoBN', 'scenario': 'standard_no_bn'},
            {'name': '2.Bottleneck', 'scenario': 'bottleneck', 'bottleneck_width': 2},
            {'name': '3.Deep', 'scenario': 'deep', 'num_layers': 12, 'lr': 0.0003},
            {'name': '4.Noisy', 'scenario': 'noisy', 'pixel_noise': 0.15, 'label_noise': 0.1, 'samples': 300},
        ]
    },
}
PFN_INTERVAL = 30  # 更频繁介入（联合优化需要更细粒度）
PFN_WARMUP_STEPS = 50  # 缩短warmup
BASE_LR = 0.0001


# ==================== CHECKPOINT MANAGER ====================
class CheckpointManager:
    """Checkpoint保存和恢复机制"""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_name(self, dataset: str, mode: str, scenario: str, batch_size: int, epoch: int) -> str:
        """生成checkpoint文件名: dataset_baseline/pfn_scenario_batch_epoch.pt"""
        return f"{dataset}_{mode}_{scenario}_b{batch_size}_epoch{epoch}"
    
    def _get_checkpoint_path(self, dataset: str, mode: str, scenario: str, batch_size: int, epoch: int) -> str:
        """获取checkpoint完整路径"""
        name = self._get_checkpoint_name(dataset, mode, scenario, batch_size, epoch)
        return os.path.join(self.checkpoint_dir, f"{name}.pt")
    
    def save_checkpoint(self, dataset: str, mode: str, scenario: str, epoch: int,
                       model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler, current_epoch: int, loss_hist: List[float], acc_hist: List[float], batch_size: int):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'loss_hist': loss_hist,
            'acc_hist': acc_hist,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self._get_checkpoint_path(dataset, mode, scenario, batch_size, epoch)
        torch.save(checkpoint, path)
        print(f"  [Checkpoint] Saved at epoch {epoch+1} to {path}")
    
    def find_latest_checkpoint(self, dataset: str, mode: str, scenario: str, batch_size: int) -> Optional[Tuple[str, int]]:
        """查找最新的checkpoint，返回路径和epoch数"""
        checkpoint_prefix = f"{dataset}_{mode}_{scenario}_b{batch_size}_epoch"
        
        import re
        latest_epoch = -1
        latest_path = None
        
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                match = re.match(f"{re.escape(checkpoint_prefix)}(\\d+)\\.pt", filename)
                if match:
                    epoch = int(match.group(1))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_path = os.path.join(self.checkpoint_dir, filename)
        
        return (latest_path, latest_epoch) if latest_path else None
    
    def load_checkpoint(self, path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler):
        """加载checkpoint（更鲁棒：支持 torch.load(..., weights_only=False)）"""
        try:
            # Prefer explicit full loading on modern PyTorch where needed
            try:
                checkpoint = torch.load(path, weights_only=False)
            except TypeError:
                # Older PyTorch doesn't accept weights_only
                checkpoint = torch.load(path)
        except Exception as e:
            print(f"  [Checkpoint] Failed to load {path}: {e}")
            # Return safe defaults so training can continue from scratch
            return 0, 0, [], []
        
        # Support both: a dict with keys ('model_state', ...) or a plain state_dict
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model_state = checkpoint.get('model_state')
            optimizer_state = checkpoint.get('optimizer_state', {})
            scheduler_state = checkpoint.get('scheduler_state', {})
            epoch = checkpoint.get('epoch', 0)
            loss_hist = checkpoint.get('loss_hist', [])
            acc_hist = checkpoint.get('acc_hist', [])
        else:
            # Treat checkpoint as a raw state_dict
            model_state = checkpoint
            optimizer_state = {}
            scheduler_state = {}
            epoch = 0
            loss_hist = []
            acc_hist = []
        
        # Apply states where available, but don't crash on partial failures
        try:
            if model_state:
                model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
        except Exception as e:
            print(f"  [Checkpoint] Warning: partial load issue for {path}: {e}")
        
        print(f"  [Checkpoint] Loaded from {path} (epoch={epoch})")
        return epoch, 0, loss_hist, acc_hist


checkpoint_manager = CheckpointManager()


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
    
    # 候选路径：Kaggle input -> 本地 data -> Kaggle working (可写)
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
            # 最后尝试下载
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
            # 最后尝试下载
            ds = DS(root=writable_root, train=train, download=True,
                   transform=(train_transform if train else test_transform))
            print(f"    Downloaded {dataset_name.upper()} to {writable_root}")
            return ds
        
        train_ds = try_load_cifar(True)
        test_ds = try_load_cifar(False)
    
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
    """
    移除checkpoint_callback参数，改为在train_model中按epoch保存
    """
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
            
            # 获取层宽度信息
            layer_widths = model.get_layer_widths() if hasattr(model, 'get_layer_widths') else None
            
            pfn['gb'].current_epoch = epoch
            pfn['gb'].total_epochs = total_epochs
            
            cap, meta = pfn['gb'].build_graph(grads, layer_widths=layer_widths)
            max_flow, cuts, S, T = pfn['solver'].find_min_cut(cap, pfn['gb'].source, pfn['gb'].sink)
            
            total_cap = sum(cap[u][v] for u, v in cuts) if cuts else 1.0
            flow_deficit = (total_cap - max_flow) / (total_cap + 1e-9)
            
            flow_dict = dict(pfn['solver'].flow) if hasattr(pfn['solver'], 'flow') else {}
            
            pfn['opt'].update_learning_rates(S, T, cuts, cap, pfn['gb'], flow_deficit,
                                            flow_dict=flow_dict, max_flow=max_flow, epoch=epoch)
        
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


def train_model(config: dict, device, use_pfn=False, logger: Optional[PFNLogger] = None):
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
    
    if use_pfn:
        groups = model.get_parameter_groups()
        for g in groups: g['lr'] = lr
        opt = optim.Adam(groups, lr=lr)
        pfn = {
            'gb': PFNGraphBuilder(history_size=5),
            'solver': IncrementalPushRelabel(),
            'opt': BottleneckOptimizer(opt, lr, base_boost=1.8, max_boost=5.0, decay_factor=0.90, lr_momentum=0.75, logger=logger)
        }
        pfn['gb'].debug = (config['epochs'] <= 20)
        pfn['opt'].debug = (config['epochs'] <= 20)
        mode = 'pfn'
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
        pfn = None
        mode = 'baseline'
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'])
    step = 0
    loss_hist, acc_hist = [], []
    acc = 0.0  # ensure defined even if training loop doesn't run (preloaded checkpoint / zero epochs)
    
    # 新增：查找和加载已有的checkpoint
    checkpoint_result = checkpoint_manager.find_latest_checkpoint(
        config['dataset'], mode, config['scenario'], config['batch_size']
    )
    
    start_epoch = 0
    if checkpoint_result:
        ckpt_path, latest_epoch = checkpoint_result
        start_epoch, step, loss_hist, acc_hist = checkpoint_manager.load_checkpoint(
            ckpt_path, model, opt, scheduler
        )
        start_epoch += 1  # 从下一个epoch开始
        # if checkpoint loaded and already at final epoch, use last acc from history
        if start_epoch > config['epochs'] and acc_hist:
            acc = acc_hist[-1]
    
    for ep in range(start_epoch, config['epochs']):
        loss, step = train_epoch(model, train_loader, opt, nn.CrossEntropyLoss(), device, 
                                pfn, step, ep, config['epochs'])
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        loss_hist.append(loss)
        acc_hist.append(acc)
        print(f"    [{ep+1:2d}/{config['epochs']}] loss={loss:.4f} acc={acc:.4f}")
        
        # 修改：每50个epoch保存一次checkpoint
        if (ep + 1) % 50 == 0:
            checkpoint_manager.save_checkpoint(
                config['dataset'], mode, config['scenario'], ep,
                model, opt, scheduler, ep, loss_hist, acc_hist, config['batch_size']
            )
    
    return acc, loss_hist, acc_hist


def run_scenario(dataset, cfg, epochs, batch, device, results_dir, logger: Optional[PFNLogger] = None):
    name = cfg['name']
    os.makedirs(os.path.join(results_dir, name), exist_ok=True)
    
    config = {
        'dataset': dataset, 'scenario': cfg['scenario'],
        'epochs': cfg.get('epochs', epochs), 'batch_size': cfg.get('batch_size', batch),
        'lr': cfg.get('lr', BASE_LR), 'bottleneck_width': cfg.get('bottleneck_width', 8),
        'num_layers': cfg.get('num_layers', 10), 'pixel_noise': cfg.get('pixel_noise', 0),
        'label_noise': cfg.get('label_noise', 0), 'samples': cfg.get('samples')
    }
    
    # Baseline: always train (no caching)
    print("  [Baseline] (训练)")
    base_acc, base_loss, base_acc_hist = train_model(config, device, False)
    
    print(f"  [PFN]")
    pfn_acc, pfn_loss, pfn_acc_hist = train_model(config, device, True, logger=logger)
    
    # 保存日志和瓶颈分析
    if logger:
        logger.save_logs(dataset, cfg['scenario'])
        _generate_bottleneck_visualization(logger, dataset, cfg['scenario'], results_dir, name)
    
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


def _generate_bottleneck_visualization(logger: PFNLogger, dataset: str, scenario: str, results_dir: str, scenario_name: str):
    """生成按epoch的瓶颈位置可视化"""
    try:
        import matplotlib.pyplot as plt
        
        if not logger.epoch_logs:
            return
        
        # 收集各epoch的关键割信息
        epochs = sorted(logger.epoch_logs.keys())
        epoch_bottlenecks = {}
        
        for epoch in epochs:
            epoch_data = logger.epoch_logs[epoch]
            critical_cuts = epoch_data['critical_cuts']
            
            # 统计该epoch的关键割位置
            bottleneck_location = {}
            for cut in critical_cuts:
                location = f"{cut['from']}->{cut['to']}"
                bottleneck_location[location] = bottleneck_location.get(location, 0) + 1
            
            # 找出最常出现的关键割
            if bottleneck_location:
                top_location = sorted(bottleneck_location.items(), key=lambda x: x[1], reverse=True)[0]
                epoch_bottlenecks[epoch] = top_location[0]
        
        if not epoch_bottlenecks:
            return
        
        # 创建可视化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 上图：每个epoch的关键割位置（文本显示）
        ax = axes[0]
        ax.axis('off')
        
        title_text = f"Critical Bottleneck Location Evolution - {scenario}\n"
        ax.text(0.05, 0.95, title_text, transform=ax.transAxes, fontsize=14, 
                fontweight='bold', verticalalignment='top', family='monospace')
        
        y_pos = 0.88
        for epoch in sorted(epoch_bottlenecks.keys()):
            bottleneck_str = f"Epoch {epoch:2d}: {epoch_bottlenecks[epoch]}"
            ax.text(0.05, y_pos, bottleneck_str, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            y_pos -= 0.05
        
        # 下图：瓶颈节点出现频率
        ax = axes[1]
        bottleneck_nodes = {}
        for epoch in epochs:
            epoch_data = logger.epoch_logs[epoch]
            for cut in epoch_data['critical_cuts']:
                node = cut['to']
                bottleneck_nodes[node] = bottleneck_nodes.get(node, 0) + 1
        
        if bottleneck_nodes:
            top_nodes = sorted(bottleneck_nodes.items(), key=lambda x: x[1], reverse=True)[:10]
            nodes = [item[0] for item in top_nodes]
            counts = [item[1] for item in top_nodes]
            
            ax.barh(nodes, counts, color='steelblue')
            ax.set_xlabel('Appearance Count in Critical Cuts', fontsize=12)
            ax.set_title('Top 10 Bottleneck Nodes', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        viz_file = os.path.join(results_dir, scenario_name, 'bottleneck_location_by_epoch.png')
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [Visualization] Saved to {viz_file}")
        
        # 打印关键割位置总结
        print(f"\n  [Critical Cuts Summary for {scenario}]")
        for epoch in sorted(epoch_bottlenecks.keys()):
            print(f"    Epoch {epoch:3d}: {epoch_bottlenecks[epoch]}")
        
    except Exception as e:
        print(f"  [Visualization] Failed: {e}")


def run_experiment(dataset='mnist', force_cpu=False, scenarios=None,
                   epochs_override=None, batch_override=None):
    device = get_device(force_cpu)
    cfg = CONFIG[dataset]
    epochs = epochs_override if epochs_override is not None else cfg['epochs_list'][0]
    batch = batch_override if batch_override is not None else cfg['batches_list'][0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'./results_{timestamp}/{dataset}_e{epochs}_b{batch}'
    os.makedirs(results_dir, exist_ok=True)
    
    logger = PFNLogger(log_dir='./pfn_log')
    
    print("=" * 60)
    print(f"  PFN EXPERIMENT - {dataset.upper()}")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch}")
    print(f"  Logs: {logger.get_run_dir()}")
    print(f"  Checkpoints: {checkpoint_manager.checkpoint_dir}")
    print("=" * 60)
    
    scenario_configs = cfg['scenarios']
    if scenarios:
        scenario_configs = [s for s in scenario_configs if s['name'] in scenarios or any(kw in s['name'].lower() for kw in scenarios)]
    
    results = []
    for i, scenario_cfg in enumerate(scenario_configs):
        print(f"\n[{i+1}/{len(scenario_configs)}] {scenario_cfg['name']}")
        print("-" * 40)
        results.append(run_scenario(dataset, scenario_cfg, epochs, batch, device, results_dir, logger))
        print(f"  >> Base={results[-1]['baseline']:.4f} | PFN={results[-1]['pfn']:.4f} | Δ={results[-1]['improvement']:+.4f}")
    
    print("\n" + "=" * 60)
    wins = sum(1 for r in results if r['improvement'] > 0)
    avg = sum(r['improvement'] for r in results) / len(results) if results else 0
    for r in results:
        print(f"{r['name']:<20} {r['baseline']:.4f}  {r['pfn']:.4f}  {r['improvement']:+.4f} {'OK' if r['improvement'] > 0 else 'X'}")
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
    parser.add_argument('--clear-ckpt', action='store_true',
                        help='Clear all checkpoints before running')
    
    args = parser.parse_args()
    
    # 处理--clear-ckpt选项
    if args.clear_ckpt:
        import shutil
        if os.path.exists(checkpoint_manager.checkpoint_dir):
            shutil.rmtree(checkpoint_manager.checkpoint_dir)
        os.makedirs(checkpoint_manager.checkpoint_dir, exist_ok=True)
        print(f"[Checkpoint] Cleared {checkpoint_manager.checkpoint_dir}")
    
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