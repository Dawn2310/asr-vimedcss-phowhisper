"""Condense long training/eval logs for the presentation notebooks (notebooks/show).

Only noise is dropped: progress bars, microbatch ticks, repeated deprecation warnings and per-clip counters.
Loss lines are thinned to every `loss_every`-th entry; every eval result, metric summary and error line is kept.
The full text stays in the original notebooks (notebooks/local) and in local/logs.
"""
import re

_BAR = re.compile(r'^\s*\d+%\|[^|]*\|\s*\d+/\d+\s*\[[^\]]*\]')
_TRAIN_PCT = re.compile(r'^\[TRAIN\] \d+/\d+ \(([\d.]+)%\)')
_COUNTER = re.compile(r'^\d+/\d+$')
_NOISE = ('microbatch', 'Trainer.tokenizer is now deprecated', 'is deprecated and will be removed')


def condense(text, loss_every=10, json_head=10):
    out, dropped = [], 0
    pct_seen, n_loss, in_json, json_lines, last_counter, last_ckpt = -1, 0, False, 0, None, None
    for raw in re.split(r'[\r\n]+', text):
        line = _BAR.sub('', raw).rstrip()
        s = line.strip()
        if not s:
            continue
        if any(k in s for k in _NOISE):
            dropped += 1
            continue
        if _COUNTER.match(s):  # per-clip eval counter: keep only the last one
            last_counter, dropped = s, dropped + 1
            continue
        if last_counter:
            out.append(last_counter)
            last_counter, dropped = None, dropped - 1
        if s.startswith('[CHECKPOINT]'):  # one per save; keep only the last, in place
            last_ckpt, dropped = (len(out), line), dropped + 1
            continue
        m = _TRAIN_PCT.match(s)
        if m:
            p = int(float(m.group(1)) // 10)
            if p == pct_seen:
                dropped += 1
                continue
            pct_seen = p
        if s.startswith(('[METRICS] {"eval_', '[METRICS] {"loss"', '[RUNNING]')):  # duplicates of the dict lines; heartbeats
            dropped += 1
            continue
        if s.startswith("{'loss'"):
            n_loss += 1
            if (n_loss - 1) % loss_every:
                dropped += 1
                continue
        if s == '{' and not in_json:
            in_json, json_lines = True, 0
        if in_json:
            json_lines += 1
            if s == '}' and line.startswith('}'):
                in_json = False
            elif json_lines > json_head:
                if json_lines == json_head + 1:
                    out.append('  …')
                dropped += 1
                continue
        out.append(line)
    if last_counter:
        out.append(last_counter)
    if last_ckpt:
        out.insert(*last_ckpt)
        dropped -= 1
    if dropped:
        out.append(f'[rút gọn để trình bày: ẩn {dropped:,} dòng tiến độ/cảnh báo lặp; loss in mỗi {loss_every} lần ghi. '
                   'Log đầy đủ: notebook gốc trong notebooks/local và local/logs]'.replace(',', '.'))
    return '\n'.join(out) + '\n'
