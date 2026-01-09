"""
日志解析脚本：将PFN实验日志转换为CSV格式
用法: python log_parser.py <log_file.txt> [output_file.csv] [--debug]
行为：
 - 逐行读取日志
 - 遇到包含 dataset/epochs/batch 的行 -> 更新 dataset, batch_size
 - 遇到 scenario header ([i/N] NAME) 且 N <= 20 -> 更新 scenario
 - 遇到 [Baseline] / [PFN] -> 更新 mode
 - 遇到 epoch 行 ([ e/Epochs ] loss=.. acc=..) -> 写一行 CSV：dataset,batch_size,scenario,mode,epoch,loss,acc
"""
import re, sys, csv
from pathlib import Path
from typing import Optional
import unicodedata

class SimpleLogParser:
    def __init__(self, infile: str, outfile: Optional[str] = None, debug: bool = False):
        self.infile = Path(infile)
        self.outfile = Path(outfile) if outfile else self.infile.with_name(self.infile.stem + '_parsed.csv')
        self.debug = debug
        # patterns
        # be tolerant to prefixes like ">> Running ..." and leading whitespace on scenario lines
        self.re_running = re.compile(r'\bRunning\b.*?dataset=(\w+)\s+epochs=(\d+)\s+batch=(\d+)', re.IGNORECASE)
        self.re_scenario = re.compile(r'^\s*\[(\d+)\s*/\s*(\d+)\]\s*(.+)$')
        self.re_baseline = re.compile(r'\[Baseline\]', re.IGNORECASE)
        self.re_pfn = re.compile(r'\[PFN\]', re.IGNORECASE)
        # more flexible epoch patterns: support floats, integers and scientific notation,
        # and allow arbitrary characters between fields (robust to extra text)
        num = r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?'
        self.re_epoch_flex = re.compile(
            rf'\[\s*(\d+)\s*/\s*\d+\].*?loss\s*[:=]\s*({num}).*?acc\s*[:=]\s*({num})',
            re.IGNORECASE)
        # legacy-ish pattern for searching fields separately (kept as fallback)
        self.re_epoch = re.compile(r'\[\s*(\d+)\s*/\s*\d+\]', re.IGNORECASE)

    def parse_and_write(self):
        if not self.infile.exists():
            print(f"Error: '{self.infile}' not found")
            return False

        dataset = None
        batch_size = None
        scenario = 'global'
        mode = 'unknown'
        total_records = 0
        per_group = {}
        bracket_lines = 0
        loss_acc_lines = 0
        used_encoding = None

        # Prepare output file: append if exists and non-empty, otherwise create and write header
        append_mode = self.outfile.exists() and self.outfile.stat().st_size > 0
        open_mode = 'a' if append_mode else 'w'
        if self.debug:
            print(f"OUTPUT MODE: {'append' if append_mode else 'write (new)'} -> {self.outfile}")

        # Read entire input file and decode once (more robust than per-line heuristics)
        with open(self.infile, 'rb') as fin, open(self.outfile, open_mode, newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            if not append_mode:
                writer.writerow(['dataset', 'batch_size', 'scenario', 'mode', 'epoch', 'loss', 'acc'])

            raw_all = fin.read()
            if not raw_all:
                if self.debug:
                    print("DEBUG: input file is empty")
            # Detect encoding: BOM preferred, otherwise heuristic on NUL bytes, then try utf-8/utf-8-sig/latin-1
            text = None
            enc = None
            # BOM checks
            if raw_all.startswith(b'\xff\xfe') or raw_all.startswith(b'\xfe\xff'):
                try:
                    text = raw_all.decode('utf-16')
                    enc = 'utf-16'
                except Exception:
                    text = None
            # heuristic UTF-16 (no BOM)
            if text is None and raw_all.count(b'\x00') > max(1, len(raw_all) // 6):
                zeros_even = sum(1 for i in range(0, len(raw_all), 2) if raw_all[i] == 0)
                zeros_odd = sum(1 for i in range(1, len(raw_all), 2) if raw_all[i] == 0)
                try:
                    if zeros_even > zeros_odd:
                        text = raw_all.decode('utf-16-be')
                        enc = 'utf-16-be'
                    else:
                        text = raw_all.decode('utf-16-le')
                        enc = 'utf-16-le'
                except Exception:
                    text = None
            # fallback attempts
            if text is None:
                for try_enc in ('utf-8-sig', 'utf-8', 'latin-1'):
                    try:
                        text = raw_all.decode(try_enc)
                        enc = try_enc
                        break
                    except Exception:
                        text = None
            if text is None:
                text = raw_all.decode('utf-8', errors='ignore')
                enc = 'utf-8 (ignore)'

            used_encoding = enc
            # normalize and strip stray NULs
            text = unicodedata.normalize('NFKC', text).replace('\u00A0', ' ').replace('\x00', '')
            lines = text.splitlines()
            if self.debug:
                print(f"DEBUG: decoded input using '{used_encoding}', total lines read: {len(lines)}")

            for line in lines:
                total_lines = None  # placeholder if you want per-line count later
                orig = line
                line = line.strip()
                if not line:
                    continue

                # debug: print info for lines that look relevant
                if self.debug and ('loss' in line or 'acc' in line or '[' in line):
                    if 'loss' in line or 'acc' in line:
                        loss_acc_lines += 1
                    print("DEBUG_LINE: repr / sample-chars:")
                    print(f"  REPR: {repr(line)}")
                    print(f"  USED-ENCODING: {used_encoding}")
                    bi = line.find('[')
                    if bi >= 0:
                        seg = line[bi:bi+40]
                        print("  BRACKET SEG:", seg)
                        print("  CODEPOINTS:", [hex(ord(c)) for c in seg])
                    else:
                        print("  no '[' found in this line")

                # Running header may appear as ">> Running ..." or "Running ..."
                m = self.re_running.search(line)
                if m:
                    dataset = m.group(1)
                    batch_size = int(m.group(3))
                    if self.debug:
                        print(f"UPDATE dataset={dataset} batch_size={batch_size}")
                    continue

                # scenario header candidate [i/N] Name  (treat as scenario if N small)
                m = self.re_scenario.match(line)
                if m:
                    idx = int(m.group(1))
                    total = int(m.group(2))
                    name = m.group(3).strip().splitlines()[0]
                    if total <= 20:
                        scenario = name
                        mode = 'unknown'  # reset until we see [Baseline] or [PFN]
                        if self.debug:
                            print(f"UPDATE scenario={scenario} (idx {idx}/{total})")
                        continue

                # mode markers
                if self.re_baseline.search(line):
                    mode = 'baseline'
                    if self.debug:
                        print(f"UPDATE mode=baseline (scenario={scenario})")
                    continue
                if self.re_pfn.search(line):
                    mode = 'pfn'
                    if self.debug:
                        print(f"UPDATE mode=pfn (scenario={scenario})")
                    continue

                # 1) try a single flexible regex capture first
                m_f = self.re_epoch_flex.search(line)
                if m_f:
                    epoch = int(m_f.group(1))
                    loss = float(m_f.group(2))
                    acc = float(m_f.group(3))
                    writer.writerow([dataset or 'unknown', batch_size or 0, scenario, mode, epoch, loss, acc])
                    total_records += 1
                    key = (scenario, mode)
                    per_group[key] = per_group.get(key, 0) + 1
                    continue

                # debug: log lines that look like epoch lines to help diagnose why they don't match
                if self.debug and ('loss' in line or 'acc' in line):
                    print("DEBUG: epoch-like line encountered:")
                    print(f"  LINE: {line}")
                    print(f"  re_epoch_flex match: {bool(m_f)}")
                    m_ep_dbg = self.re_epoch.search(line)
                    m_loss_dbg = re.search(r'loss\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)', line, re.IGNORECASE)
                    m_acc_dbg = re.search(r'acc\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)', line, re.IGNORECASE)
                    print(f"  epoch found: {bool(m_ep_dbg)} (val: {m_ep_dbg.group(1) if m_ep_dbg else 'N/A'})")
                    print(f"  loss found: {bool(m_loss_dbg)} (val: {m_loss_dbg.group(1) if m_loss_dbg else 'N/A'})")
                    print(f"  acc found: {bool(m_acc_dbg)} (val: {m_acc_dbg.group(1) if m_acc_dbg else 'N/A'})")
                    continue

                # 2) fallback: look for fields separately (more permissive numeric regexp)
                if 'loss' in line and 'acc' in line:
                    m_ep = self.re_epoch.search(line)
                    # accept separators like "loss=1.23", "loss:1.23", "loss 1.23" etc.
                    m_loss = re.search(r'loss[^0-9+-]*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)', line, re.IGNORECASE)
                    m_acc = re.search(r'acc[^0-9+-]*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)', line, re.IGNORECASE)

                    if m_loss and m_acc:
                        epoch = int(m_ep.group(1)) if m_ep else -1
                        loss = float(m_loss.group(1))
                        acc = float(m_acc.group(1))
                        writer.writerow([dataset or 'unknown', batch_size or 0, scenario, mode, epoch, loss, acc])
                        total_records += 1
                        key = (scenario, mode)
                        per_group[key] = per_group.get(key, 0) + 1
                        continue
                    else:
                        if self.debug:
                            print(f"Unparsed epoch-like line (missing fields): {line}")

                # Debug: detect any bracketed epoch token even if loss/acc not matched
                if self.re_epoch.search(line):
                    bracket_lines += 1
                    if self.debug:
                        print(f"DEBUG: bracketed epoch line (dataset={dataset or 'unknown'} scenario={scenario} mode={mode}):")
                        print(f"  {repr(line)}")
                        print(f"  has 'loss': {'loss' in line}, has 'acc': {'acc' in line}, used_encoding={used_encoding}")

        if append_mode:
            print(f"✓ Successfully appended {total_records} records to {self.outfile}")
        else:
            print(f"✓ Successfully saved to {self.outfile}")
            print(f"  Total records: {total_records}")
        if append_mode and total_records and self.debug:
            print(f"  (Appended {total_records} new rows)")
        if total_records == 0:
            print("  注意：未解析到任何记录。可能原因：日志中的 epoch/loss/acc 格式与预期不完全一致。")
            print("         请用 --debug 重新运行以查看未匹配的行，或把包含 epoch/loss/acc 的几行示例贴上来以便改进解析器。")
        if self.debug:
            print(f"  DEBUG: used decoding: {used_encoding}")
            print(f"  DEBUG: loss/acc-like lines seen: {loss_acc_lines}")
            print(f"  DEBUG: bracketed epoch lines seen: {bracket_lines}")
            print("  Breakdown:")
            for (sc, md), cnt in sorted(per_group.items()):
                print(f"    {sc} | {md}: {cnt}")
        return True


def main():
    debug = '--debug' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--debug']
    if not args:
        print("Usage: python log_parser.py <log_file> [output.csv] [--debug]")
        return
    infile = args[0]
    outfile = args[1] if len(args) > 1 else None

    parser = SimpleLogParser(infile, outfile, debug=debug)
    ok = parser.parse_and_write()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
