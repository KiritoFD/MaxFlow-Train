import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).parent
OUTPUT_JSON = BASE_DIR / "merged_results.json"
OUTPUT_CSV = BASE_DIR / "merged_results.csv"

def load_summaries() -> List[tuple[Path, Dict[str, Any]]]:
    """Load all *summary.json files and return (file_path, data) pairs."""
    summaries = []
    for summary_file in BASE_DIR.rglob("*summary.json"):
        try:
            with summary_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            summaries.append((summary_file, data))
        except (json.JSONDecodeError, IOError):
            continue
    return summaries

def extract_metadata_from_path(file_path: Path) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Extract dataset, epoch, batch_size from path like 'mnist_e30_b128' or 'cifar100_e100_b512'."""
    dataset, epoch, batch_size = None, None, None
    
    # Search through path parts for pattern like {dataset}_e{epoch}_b{batch}
    for part in file_path.parts:
        m = re.match(r"^([a-z0-9]+)_e(\d+)_b(\d+)$", part)
        if m:
            dataset = m.group(1)
            epoch = int(m.group(2))
            batch_size = int(m.group(3))
            break
    
    return dataset, epoch, batch_size

def sample_acc_values(acc_list: List[float], target_count: int = 20) -> List[Optional[float]]:
    """Sample acc values evenly across the list."""
    if not acc_list:
        return [None] * target_count
    if len(acc_list) <= target_count:
        return acc_list + [None] * (target_count - len(acc_list))
    step = (len(acc_list) - 1) / (target_count - 1)
    return [acc_list[int(i * step)] for i in range(target_count)]

def load_loss_json(file_path: Path) -> List[float]:
    """Load acc array from loss JSON file."""
    try:
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("acc", [])
    except (json.JSONDecodeError, IOError):
        pass
    return []

def load_log_json(file_path: Path) -> tuple[List[float], List[float]]:
    """Load baseline and pfn acc arrays from log.json file."""
    baseline_acc = []
    pfn_acc = []
    try:
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                baseline_acc = data.get("baseline", {}).get("acc", [])
                pfn_acc = data.get("pfn", {}).get("acc", [])
    except (json.JSONDecodeError, IOError):
        pass
    return baseline_acc, pfn_acc

def collect_all_records() -> List[Dict[str, Any]]:
    """Collect records from summaries and also scan for orphan log/loss files."""
    all_records = []
    processed_paths = set()
    
    summaries = load_summaries()
    
    # Process summaries
    for summary_file, summary_data in summaries:
        summary_dir = summary_file.parent
        
        # Extract from summary first, then from path
        dataset = summary_data.get("dataset")
        epoch = summary_data.get("epochs")
        batch_size = summary_data.get("batch_size")
        
        # Try to extract from path if summary data is incomplete
        if not dataset or not epoch or not batch_size:
            path_dataset, path_epoch, path_batch = extract_metadata_from_path(summary_file)
            dataset = dataset or path_dataset
            epoch = epoch or path_epoch
            batch_size = batch_size or path_batch
        
        results = summary_data.get("results", [])
        for result in results:
            cond_name = result.get("name", "")
            m = re.match(r"^([0-3])\.", cond_name)
            cond_code = int(m.group(1)) if m else None
            cond_display = cond_name.split(".", 1)[1] if "." in cond_name else cond_name
            
            baseline_file = summary_dir / f"{cond_name}" / "baseline_loss.json"
            pfn_file = summary_dir / f"{cond_name}" / "pfn_loss.json"
            log_file = summary_dir / f"{cond_name}" / "log.json"
            
            baseline_acc = load_loss_json(baseline_file)
            pfn_acc = load_loss_json(pfn_file)
            
            # Try log.json if loss files are empty
            if not baseline_acc or not pfn_acc:
                baseline_acc_log, pfn_acc_log = load_log_json(log_file)
                baseline_acc = baseline_acc or baseline_acc_log
                pfn_acc = pfn_acc or pfn_acc_log
            
            baseline_sampled = sample_acc_values(baseline_acc, 20)
            pfn_sampled = sample_acc_values(pfn_acc, 20)
            
            record = {
                "dataset": dataset,
                "epoch": epoch,
                "batch_size": batch_size,
                "condition_code": cond_code,
                "condition_name": cond_display,
                "summary_source": str(summary_file.relative_to(BASE_DIR)),
                "baseline_acc": baseline_sampled,
                "pfn_acc": pfn_sampled,
                "baseline_final": baseline_acc[-1] if baseline_acc else None,
                "pfn_final": pfn_acc[-1] if pfn_acc else None,
            }
            all_records.append(record)
            processed_paths.add(baseline_file.resolve())
            processed_paths.add(pfn_file.resolve())
            processed_paths.add(log_file.resolve())
    
    # Scan for orphan log.json files (not covered by summaries)
    for log_file in BASE_DIR.rglob("log.json"):
        if log_file.resolve() in processed_paths:
            continue
        
        baseline_acc, pfn_acc = load_log_json(log_file)
        if not baseline_acc and not pfn_acc:
            continue
        
        # Extract metadata from path
        dataset, epoch, batch_size = extract_metadata_from_path(log_file)
        cond_code, cond_name = None, None
        for part in log_file.parts:
            m = re.match(r"^([0-3])\.(.+)$", part)
            if m:
                cond_code = int(m.group(1))
                cond_name = m.group(2)
                break
        
        baseline_sampled = sample_acc_values(baseline_acc, 20)
        pfn_sampled = sample_acc_values(pfn_acc, 20)
        
        record = {
            "dataset": dataset,
            "epoch": epoch,
            "batch_size": batch_size,
            "condition_code": cond_code,
            "condition_name": cond_name,
            "summary_source": None,
            "baseline_acc": baseline_sampled,
            "pfn_acc": pfn_sampled,
            "baseline_final": baseline_acc[-1] if baseline_acc else None,
            "pfn_final": pfn_acc[-1] if pfn_acc else None,
        }
        all_records.append(record)
        processed_paths.add(log_file.resolve())
    
    return all_records

def main() -> None:
    all_records = collect_all_records()
    
    # Write JSON
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    
    # Write CSV
    csv_headers = [
        "dataset", "epoch", "batch_size", "condition_code", "condition_name",
        "summary_source", "baseline_final", "pfn_final"
    ]
    csv_headers += [f"baseline_acc_{i+1}" for i in range(20)]
    csv_headers += [f"pfn_acc_{i+1}" for i in range(20)]
    
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for record in all_records:
            row = {
                "dataset": record["dataset"],
                "epoch": record["epoch"],
                "batch_size": record["batch_size"],
                "condition_code": record["condition_code"],
                "condition_name": record["condition_name"],
                "summary_source": record["summary_source"],
                "baseline_final": record["baseline_final"],
                "pfn_final": record["pfn_final"],
            }
            for i, val in enumerate(record["baseline_acc"]):
                row[f"baseline_acc_{i+1}"] = val
            for i, val in enumerate(record["pfn_acc"]):
                row[f"pfn_acc_{i+1}"] = val
            writer.writerow(row)
    
    print(f"Processed {len(all_records)} records")
    print(f"JSON output: {OUTPUT_JSON}")
    print(f"CSV output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
