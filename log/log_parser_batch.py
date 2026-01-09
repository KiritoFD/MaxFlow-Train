"""
批量日志解析脚本：将多个PFN实验日志转换为CSV格式
用法: python log_parser_batch.py [log_dir] [output_file.csv]
"""

import re
import sys
import csv
from pathlib import Path
from typing import List, Dict
from log_parser import LogParser


def batch_parse(log_dir: str = '.', output_file: str = 'all_results.csv'):
    """批量解析日志目录下的所有txt文件"""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Error: Directory '{log_dir}' not found")
        return
    
    # 查找所有.log或.txt文件
    log_files = list(log_path.glob('*.log')) + list(log_path.glob('*.txt'))
    
    if not log_files:
        print(f"Error: No log files found in '{log_dir}'")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    all_records = []
    
    for log_file in log_files:
        print(f"\nParsing: {log_file.name}")
        try:
            parser = LogParser(str(log_file))
            records = parser.parse()
            all_records.extend(records)
            print(f"  ✓ {len(records)} records extracted")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if not all_records:
        print("Error: No records extracted from any log files")
        return
    
    # 保存合并结果
    fieldnames = ['dataset', 'batch_size', 'scenario', 'mode', 'epoch', 'loss', 'acc']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        
        print(f"\n✓ Successfully saved to {output_file}")
        print(f"  Total records: {len(all_records)}")
        
        # 打印统计信息
        print("\nStatistics:")
        datasets = set(r['dataset'] for r in all_records)
        scenarios = set(r['scenario'] for r in all_records)
        modes = set(r['mode'] for r in all_records)
        
        print(f"  Datasets: {sorted(datasets)}")
        print(f"  Scenarios: {len(scenarios)}")
        print(f"  Modes: {sorted(modes)}")
        
    except Exception as e:
        print(f"Error: Failed to save CSV - {e}")


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'all_results.csv'
    
    print(f"Batch parsing logs from: {log_dir}")
    print(f"Output file: {output_file}")
    
    batch_parse(log_dir, output_file)


if __name__ == '__main__':
    main()
