import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "merged_results.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palettes: baseline = blue shades, pfn = red shades
# Order: smallest batch = darkest color (index 0), largest batch = lightest color (index 4)
BASELINE_COLORS = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef']
PFN_COLORS = ['#b30000', '#e34a33', '#fc8d59', '#fdbb84', '#fdd49e']

def read_csv_data():
    """Read CSV and organize data by dataset."""
    data_by_dataset = defaultdict(lambda: defaultdict(list))
    
    with CSV_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            condition_code = row["condition_code"]
            condition_name = row["condition_name"]
            epoch = row["epoch"]
            batch_size = row["batch_size"]
            
            # Extract baseline and pfn accuracy values
            baseline_acc = []
            pfn_acc = []
            for i in range(1, 21):
                baseline_val = row.get(f"baseline_acc_{i}")
                pfn_val = row.get(f"pfn_acc_{i}")
                baseline_acc.append(float(baseline_val) if baseline_val else None)
                pfn_acc.append(float(pfn_val) if pfn_val else None)
            
            # Remove None values
            baseline_acc = [v for v in baseline_acc if v is not None]
            pfn_acc = [v for v in pfn_acc if v is not None]
            
            key = f"{condition_code}.{condition_name}_e{epoch}_b{batch_size}"
            data_by_dataset[dataset][key] = {
                "baseline": baseline_acc,
                "pfn": pfn_acc,
                "condition_code": condition_code,
                "condition_name": condition_name,
                "epoch": epoch,
                "batch_size": batch_size,
            }
    
    return data_by_dataset

def plot_dataset(dataset: str, data: dict):
    """Plot all conditions for a dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset.upper()} - Training Accuracy Curves", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    by_condition = defaultdict(list)
    for key, values in data.items():
        cond_code = int(values["condition_code"])
        by_condition[cond_code].append((key, values))
    
    for cond_code in sorted(by_condition.keys()):
        ax = axes[cond_code]
        items = by_condition[cond_code]
        cond_name = items[0][1]["condition_name"] if items else "Unknown"
        
        # Sort items by batch_size for consistent color assignment
        sorted_items = sorted(items, key=lambda x: int(x[1]["batch_size"]) if x[1]["batch_size"] else 0)
        
        for color_idx, (key, values) in enumerate(sorted_items):
            baseline = values["baseline"]
            pfn = values["pfn"]
            batch_size = values["batch_size"]
            x = list(range(len(baseline)))
            
            baseline_color = BASELINE_COLORS[min(color_idx, len(BASELINE_COLORS)-1)]
            pfn_color = PFN_COLORS[min(color_idx, len(PFN_COLORS)-1)]
            
            if baseline:
                ax.plot(x, baseline, marker='o', linewidth=2, markersize=4, color=baseline_color, alpha=0.8, label=f"Baseline (b={batch_size})")
            if pfn:
                ax.plot(x, pfn, marker='s', linewidth=2, markersize=4, color=pfn_color, alpha=0.8, linestyle='--', label=f"PFN (b={batch_size})")
        
        ax.set_xlabel("Epoch Sample (10-point sampling)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{cond_name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{dataset}_accuracy_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_comparison_by_condition(dataset: str, data: dict):
    """Plot comparison of all condition/batch/epoch combinations for a dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset.upper()} - All Conditions Comparison", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    by_condition = defaultdict(list)
    for key, values in data.items():
        cond_code = int(values["condition_code"])
        by_condition[cond_code].append((key, values))
    
    for cond_code in sorted(by_condition.keys()):
        ax = axes[cond_code]
        items = by_condition[cond_code]
        cond_name = items[0][1]["condition_name"] if items else "Unknown"
        
        # Sort items by batch_size for consistent color assignment
        sorted_items = sorted(items, key=lambda x: int(x[1]["batch_size"]) if x[1]["batch_size"] else 0)
        
        for color_idx, (key, values) in enumerate(sorted_items):
            baseline = values["baseline"]
            pfn = values["pfn"]
            batch_size = values["batch_size"]
            x = list(range(len(baseline)))
            
            baseline_color = BASELINE_COLORS[min(color_idx, len(BASELINE_COLORS)-1)]
            pfn_color = PFN_COLORS[min(color_idx, len(PFN_COLORS)-1)]
            
            if baseline:
                ax.plot(x, baseline, marker='o', label=f"Baseline (b={batch_size})", linewidth=1.5, markersize=3, color=baseline_color, alpha=0.8)
            if pfn:
                ax.plot(x, pfn, marker='s', label=f"PFN (b={batch_size})", linewidth=1.5, markersize=3, color=pfn_color, alpha=0.8, linestyle='--')
        
        ax.set_xlabel("Epoch Sample (10-point sampling)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{cond_name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{dataset}_all_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_batch_size_comparison(dataset: str, data: dict):
    """Plot comparison of different batch sizes for each condition."""
    by_condition = defaultdict(lambda: defaultdict(list))
    
    for key, values in data.items():
        cond_code = int(values["condition_code"])
        batch_size = values["batch_size"]
        by_condition[cond_code][batch_size].append((key, values))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"{dataset.upper()} - Batch Size Comparison by Condition", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for cond_code in sorted(by_condition.keys()):
        ax = axes[cond_code]
        batch_dict = by_condition[cond_code]
        cond_name = None
        
        # Sort batch sizes from small to large for color assignment
        batch_sizes = sorted(batch_dict.keys(), key=lambda x: int(x) if x else 0)
        
        for color_idx, batch_size in enumerate(batch_sizes):
            items = batch_dict[batch_size]
            baseline_color = BASELINE_COLORS[min(color_idx, len(BASELINE_COLORS)-1)]
            pfn_color = PFN_COLORS[min(color_idx, len(PFN_COLORS)-1)]
            
            for key, values in items:
                if cond_name is None:
                    cond_name = values["condition_name"]
                
                baseline = values["baseline"]
                pfn = values["pfn"]
                x = list(range(len(baseline)))
                
                if baseline:
                    ax.plot(x, baseline, marker='o', linewidth=2, markersize=4, color=baseline_color, alpha=0.8, label=f"Baseline (b={batch_size})")
                    if baseline:
                        ax.text(x[-1] + 0.3, baseline[-1], f"b={batch_size}", fontsize=9, color=baseline_color, fontweight='bold')
                
                if pfn:
                    ax.plot(x, pfn, marker='s', linewidth=2, markersize=4, color=pfn_color, alpha=0.8, linestyle='--', label=f"PFN (b={batch_size})")
                    if pfn:
                        ax.text(x[-1] + 0.3, pfn[-1], f"b={batch_size}", fontsize=9, color=pfn_color, fontweight='bold')
        
        ax.set_xlabel("Epoch Sample (10-point sampling)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{cond_name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    output_file = OUTPUT_DIR / f"{dataset}_batch_size_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_baseline_vs_pfn_by_condition(dataset: str, data: dict):
    """Plot baseline vs PFN improvement for each condition."""
    by_condition = defaultdict(list)
    
    for key, values in data.items():
        cond_code = int(values["condition_code"])
        by_condition[cond_code].append((key, values))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"{dataset.upper()} - Baseline vs PFN by Condition", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for cond_code in sorted(by_condition.keys()):
        ax = axes[cond_code]
        items = by_condition[cond_code]
        cond_name = items[0][1]["condition_name"] if items else "Unknown"
        
        # Sort items by batch_size for consistent color assignment
        sorted_items = sorted(items, key=lambda x: int(x[1]["batch_size"]) if x[1]["batch_size"] else 0)
        
        for color_idx, (key, values) in enumerate(sorted_items):
            baseline = values["baseline"]
            pfn = values["pfn"]
            batch_size = values["batch_size"]
            x = list(range(len(baseline)))
            
            baseline_color = BASELINE_COLORS[min(color_idx, len(BASELINE_COLORS)-1)]
            pfn_color = PFN_COLORS[min(color_idx, len(PFN_COLORS)-1)]
            
            if baseline:
                ax.plot(x, baseline, marker='o', linewidth=2, markersize=4, color=baseline_color, alpha=0.8, label=f"Baseline (b={batch_size})")
                if baseline:
                    ax.text(x[-1] + 0.3, baseline[-1], f"b={batch_size}", fontsize=9, color=baseline_color, fontweight='bold')
            
            if pfn:
                ax.plot(x, pfn, marker='s', linewidth=2, markersize=4, color=pfn_color, alpha=0.8, linestyle='--', label=f"PFN (b={batch_size})")
                if pfn:
                    ax.text(x[-1] + 0.3, pfn[-1], f"b={batch_size}", fontsize=9, color=pfn_color, fontweight='bold')
        
        ax.set_xlabel("Epoch Sample (10-point sampling)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{cond_name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    output_file = OUTPUT_DIR / f"{dataset}_baseline_vs_pfn.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_individual_conditions(dataset: str, data: dict):
    """Plot each condition separately as individual PNG files."""
    by_condition = defaultdict(list)
    for key, values in data.items():
        cond_code = int(values["condition_code"])
        by_condition[cond_code].append((key, values))
    
    for cond_code in sorted(by_condition.keys()):
        items = by_condition[cond_code]
        cond_name = items[0][1]["condition_name"] if items else "Unknown"
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"{dataset.upper()} - {cond_name}", fontsize=16, fontweight='bold')
        
        # Sort items by batch_size for consistent color assignment
        sorted_items = sorted(items, key=lambda x: int(x[1]["batch_size"]) if x[1]["batch_size"] else 0)
        
        for color_idx, (key, values) in enumerate(sorted_items):
            baseline = values["baseline"]
            pfn = values["pfn"]
            batch_size = values["batch_size"]
            x = list(range(len(baseline)))
            
            baseline_color = BASELINE_COLORS[min(color_idx, len(BASELINE_COLORS)-1)]
            pfn_color = PFN_COLORS[min(color_idx, len(PFN_COLORS)-1)]
            
            if baseline:
                ax.plot(x, baseline, marker='o', linewidth=2.5, markersize=5, color=baseline_color, alpha=0.8, label=f"Baseline (b={batch_size})")
            if pfn:
                ax.plot(x, pfn, marker='s', linewidth=2.5, markersize=5, color=pfn_color, alpha=0.8, linestyle='--', label=f"PFN (b={batch_size})")
        
        ax.set_xlabel("Epoch Sample (10-point sampling)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / f"{dataset}_{cond_code}_{cond_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def main():
    data_by_dataset = read_csv_data()
    
    print(f"Found {len(data_by_dataset)} datasets")
    
    for dataset in sorted(data_by_dataset.keys()):
        print(f"\nProcessing {dataset}...")
        plot_dataset(dataset, data_by_dataset[dataset])
        plot_comparison_by_condition(dataset, data_by_dataset[dataset])
        plot_batch_size_comparison(dataset, data_by_dataset[dataset])
        plot_baseline_vs_pfn_by_condition(dataset, data_by_dataset[dataset])
        plot_individual_conditions(dataset, data_by_dataset[dataset])
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
