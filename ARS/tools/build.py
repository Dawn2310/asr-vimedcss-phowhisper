"""Build the ARS folder: self-contained presentation notebooks for ASR Task 2, without retraining anything.

1. Copy the saved results the notebooks need into ARS/saved (tables, trainer states, metrics, configs, normalized
   predictions) and condense the training/eval logs (tools/show_utils.condense).
2. Write the notebooks and execute them on CPU, so every table and figure is a real output of reading saved files.

Run with the project venv:  .venv-asr/Scripts/python.exe ARS/tools/build.py
Nothing under ASR/ is modified.
"""
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ARS = Path(__file__).resolve().parents[1]
ASR = ARS.parent if (ARS.parent / 'asr').is_dir() else ARS.parent / 'ASR'  # ARS inside the ASR repo, or beside it
SAVED = ARS / 'saved'
sys.path.insert(0, str(ASR))
sys.path.insert(0, str(ARS / 'tools'))
from asr.metrics import normalize_text, split_terms  # noqa: E402
from show_utils import condense  # noqa: E402

RUNS, EXP, LOGS, NBS = ASR / 'local/runs', ASR / 'experiments', ASR / 'local/logs', ASR / 'notebooks/local'
VF = EXP / 'O3W_whisper_large_v3_lora_official_termos'


# ------------------------------------------------------------------ 1. saved files
def put(src, rel):
    dst = SAVED / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def cell_text(nb, idx):
    c = nbformat.read(NBS / nb, as_version=4).cells[idx]
    return c.source, ''.join(o.get('text', '') for o in c.get('outputs', []))


def log_of(nb, idx):
    """Raw log files named by the cell's [LOG] lines (full text), else the cell output itself."""
    out = cell_text(nb, idx)[1]
    paths = [Path(p.strip()) for p in re.findall(r'\[LOG\] (.+?\.log)', out)]
    if paths and all(p.is_file() for p in paths):
        return '\n'.join(f'===== {p.name}\n' + p.read_text(encoding='utf-8', errors='replace') for p in paths)
    return out


def write_log(name, text):
    p = SAVED / 'logs' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(condense(text), encoding='utf-8')


def predictions(src, rel):
    rows = []
    for line in src.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        rows.append(dict(id=r['id'], source=r['source'], duration=r.get('duration'), ref=normalize_text(r['text']),
                         hyp=normalize_text(r['hyp']), terms=split_terms(r.get('cs_terms') or ''), truncated=bool(r.get('truncated', False))))
    p = SAVED / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')


def collect():
    if SAVED.exists():
        shutil.rmtree(SAVED)
    for f in (ASR / 'reports/task2_20261030/tables').iterdir():
        put(f, f'tables/{f.name}')
    put(RUNS / 'V1_phowhisper_small_lora/local_v1/checkpoints/checkpoint-2724/trainer_state.json', 'runs/V1_trainer_state.json')
    put(RUNS / 'V2_phowhisper_small_lora_cont/local_v2/checkpoints/checkpoint-1816/trainer_state.json', 'runs/V2_trainer_state.json')
    put(VF / 'runs.jsonl', 'runs/VFinal_runs.jsonl')
    put(ASR / 'reports/final_freeze_official/freeze.json', 'runs/VFinal_freeze.json')
    put(ASR / 'data/v3_train_exclude.json', 'runs/train_exclude.json')
    for name, exp in [('_base', '_base'), ('Baseline1', 'B0_phowhisper_small_zeroshot'), ('Baseline2', 'B1_whisper_small_zeroshot'),
                      ('V1', 'V1_phowhisper_small_lora'), ('V2', 'V2_phowhisper_small_lora_cont'), ('VFinal', VF.name)]:
        put(EXP / f'{exp}.yaml' if exp == '_base' else EXP / exp / 'config.yaml', f'configs/{name}.yaml')
    for name, src in [('Baseline1_dev', RUNS / 'B0_phowhisper_small_zeroshot/dev_local_v1/eval/dev'),
                      ('Baseline2_dev', RUNS / 'B1_whisper_small_zeroshot/dev_local_v1/eval/dev'),
                      ('V1_dev', RUNS / 'V1_phowhisper_small_lora/dev_local_v1/eval/dev'),
                      ('V2_dev', RUNS / 'V2_phowhisper_small_lora_cont/dev_local_v2/eval/dev'),
                      ('Baseline1_test', EXP / 'B0_phowhisper_small_zeroshot/eval/test'),
                      ('Baseline1_hard', EXP / 'B0_phowhisper_small_zeroshot/eval/hard'),
                      *[(f'VFinal_{k}', VF / 'eval' / k) for k in ('dev', 'dev_beam5', 'test_beam5', 'hard_beam5')]]:
        put(src / 'metrics.json', f'metrics/{name}.json')
    (SAVED / 'runs/VFinal_adapter.json').write_text(json.dumps(
        {'adapter_MB': (VF / 'final/adapter_model.safetensors').stat().st_size / 1e6}), encoding='utf-8')  # MB = 10^6 bytes, as latency_size.csv
    predictions(RUNS / 'V1_phowhisper_small_lora/dev_local_v1/eval/dev/predictions.jsonl', 'predictions/V1_dev.jsonl')
    predictions(VF / 'eval/test_beam5/predictions.jsonl', 'predictions/VFinal_test.jsonl')
    import pyarrow.parquet as pq
    excl = set(json.loads((ASR / 'data/v3_train_exclude.json').read_text(encoding='utf-8'))['ids'])
    t = pq.read_table(ASR / 'data/ViMedCSS/train.parquet', columns=['id', 'cs_terms', 'duration']).to_pylist()
    c = Counter(x for r in t if r['id'] not in excl and 0 < r['duration'] <= 30 for x in split_terms(r['cs_terms'] or ''))
    (SAVED / 'runs/train_terms_official.json').write_text(json.dumps(dict(c), ensure_ascii=False), encoding='utf-8')

    write_log('Baseline1_dev_eval.txt', log_of('01_B0.ipynb', 5))
    write_log('Baseline2_dev_eval.txt', log_of('02_B1.ipynb', 5))
    write_log('V1_train.txt', log_of('03_V1.ipynb', 5))
    write_log('V1_dev_eval.txt', log_of('03_V1.ipynb', 8))
    write_log('V2_train.txt', '\n'.join(f'===== {f}\n' + (LOGS / f).read_text(encoding='utf-8', errors='replace')
                                        for f in ('20261023_234313_train_v2.log', '20261023_235920_599400.log')))
    write_log('V2_dev_eval.txt', (LOGS / '20261024_113508_291573.log').read_text(encoding='utf-8', errors='replace'))
    write_log('thanh_phan_chia_theo_video_train.txt', log_of('09_V3.ipynb', 7))
    write_log('thanh_phan_specaugment_train.txt', log_of('10_V4.ipynb', 5))
    write_log('thanh_phan_medium_train.txt', log_of('12_V5.ipynb', 5))


# ------------------------------------------------------------------ 2. notebooks
SETUP = '''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
S = next(p / 'saved' for p in [Path.cwd(), *Path.cwd().parents] if (p / 'saved/tables').is_dir())
T = S / 'tables'
def js(rel): return json.loads((S / rel).read_text(encoding='utf-8'))
def show_log(name): print((S / 'logs' / name).read_text(encoding='utf-8'))
def history(name): return js(f'runs/{name}_trainer_state.json')['log_history']
pd.set_option('display.precision', 2)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'Baseline 1': '#7E9C9B', 'Baseline 2': '#A9C2C0', 'V1': '#0F4E57', 'V2': '#1A6771', 'V-Final': '#FFB21A'}'''

HEAD = ('> Notebook trình bày, **không huấn luyện lại**. Mọi bảng và hình được tính từ file kết quả đã lưu trong `saved/`, '
        'chạy lại an toàn trên CPU. Log huấn luyện đã được rút gọn (bỏ thanh tiến độ, giữ mọi kết quả eval); '
        'bản đầy đủ nằm trong `ASR/notebooks/local` và `ASR/local/logs`.')

MAP = '''| Tên trình bày | Mô hình | Thư mục thí nghiệm trong ASR |
|---|---|---|
| Baseline 1 | PhoWhisper-small, chưa huấn luyện | `B0_phowhisper_small_zeroshot` |
| Baseline 2 | Whisper-small, chưa huấn luyện | `B1_whisper_small_zeroshot` |
| V1 | PhoWhisper-small + LoRA, lr 1e-3, trộn 4 nguồn | `V1_phowhisper_small_lora` |
| V2 | V1 train tiếp, lr 1e-4 | `V2_phowhisper_small_lora_cont` |
| V-Final | Whisper-large-v3 + LoRA + SpecAugment + oversampling thuật ngữ hiếm, beam 5 | `O3W_whisper_large_v3_lora_official_termos` |'''

DEV_TABLE = '''def dev_row(name):
    m = js(f'metrics/{name}.json'); o, s = m['overall'], m['by_source']
    return {'WER %': 100 * o['wer'], 'CER %': 100 * o['cer'], 'Recall thuật ngữ %': 100 * o.get('cs_term_recall', float('nan')),
            **{k: 100 * s[k]['wer'] for k in ['ViMedCSS', 'VietMed', 'FLEURS', 'VIVOS']}}
DEV = pd.DataFrame({'Baseline 1': dev_row('Baseline1_dev'), 'Baseline 2': dev_row('Baseline2_dev'), 'V1': dev_row('V1_dev'),
                    'V2': dev_row('V2_dev'), 'V-Final': dev_row('VFinal_dev_beam5')}).T
DEV'''


class Book:
    def __init__(self, name, title):
        self.name = name
        self.cells = [nbformat.v4.new_markdown_cell(title.strip()), nbformat.v4.new_markdown_cell(HEAD),
                      nbformat.v4.new_code_cell(SETUP)]

    def md(self, text):
        self.cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def code(self, text):
        self.cells.append(nbformat.v4.new_code_cell(text.strip()))

    def command(self, nb, idx, label):
        """Show the code that was actually run, as text (not executable here)."""
        src = cell_text(nb, idx)[0].strip()
        self.md(f'**Lệnh đã chạy — {label}** (chép từ `ASR/notebooks/local/{nb}`, ô {idx}; chỉ để xem)\n\n```python\n{src}\n```')

    def build(self):
        nb = nbformat.v4.new_notebook(cells=self.cells, metadata={
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}})
        NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ARS)}}).execute()
        nbformat.validate(nb)
        nbformat.write(nb, ARS / self.name)
        print(f'{self.name}: {len(nb.cells)} ô, {(ARS / self.name).stat().st_size / 1024:.0f} KB', flush=True)


def nb00():
    b = Book('00_Tong_quan.ipynb', '# 00 — Tổng quan: Nhận dạng giọng nói y khoa Việt–Anh (Task 2)')
    b.md(f'''{MAP}

| Yêu cầu Task 2 | Notebook |
|---|---|
| Thống kê nguồn, làm sạch, mất cân bằng, chia tập | `01_Du_lieu` |
| 2 baseline, so với paper, gap analysis | `02_Baseline` |
| V1: kiến trúc, đường cong loss/WER, tín hiệu cải thiện, phân tích lỗi, cặp từ nhầm | `03_V1` |
| V2: cải tiến từ phân tích lỗi V1 | `04_V2` |
| V-Final: cấu hình, TEST/HARD, so với paper, ví dụ đầu ra | `05_V_Final` |
| Thí nghiệm thành phần, tính khả thi | `06_Thanh_phan_Kha_thi` |''')
    b.md('## Kết quả trên cùng DEV chính thức (5.168 clip)')
    b.code(DEV_TABLE)
    b.code('''fig, ax = plt.subplots(figsize=(7.5, 3.3))
bars = ax.bar(DEV.index, DEV['WER %'], color=[C[k] for k in DEV.index])
ax.bar_label(bars, fmt='%.2f'); ax.set_ylabel('WER % (DEV chính thức)'); ax.set_title('Tiến trình qua các phiên bản')
plt.show()''')
    b.md('## V-Final trên TEST/HARD chính thức, cùng cách chấm với các mô hình chưa huấn luyện')
    b.code('''s = pd.read_csv(T / 'same_metric_official.csv')
lab = {'B0': 'Baseline 1 (PhoWhisper-small)', 'B2': 'PhoWhisper-large', 'B3': 'Whisper-large-v3', 'O3W': 'V-Final'}
s = s[s.system.isin(lab)].assign(system=lambda x: x.system.map(lab))
s.pivot_table(index='system', columns='split', values='wer').reindex(list(lab.values()))''')
    b.build()


def nb01():
    b = Book('01_Du_lieu.ipynb', '# 01 — Dữ liệu: nguồn, làm sạch, mất cân bằng')
    b.md('''## 1.1 Nguồn dữ liệu
Cả 4 nguồn là dữ liệu công khai (Online). Nhóm không thu dữ liệu Offline. Sau xử lý: FLAC 16 kHz mono trong parquet,
các trường `id, source, split, group_id, speaker_id, text, cs_terms, duration, audio`.''')
    b.code('''d = pd.read_csv(T / 'data_stats_by_source_split.csv')
print('Số clip (ViMedCSS_vsplit = ViMedCSS chia lại theo video, dùng cho thí nghiệm thành phần ở 06)')
display(d.pivot_table(index='source', columns='split', values='clips', aggfunc='sum', fill_value=0)[['train', 'dev', 'test', 'hard']])
print('Số giờ')
d.pivot_table(index='source', columns='split', values='hours', aggfunc='sum', fill_value=0)[['train', 'dev', 'test', 'hard']].round(1)''')
    b.code("pd.read_csv(T / 'split_ratio_official.csv').set_index('split')")
    b.md('## 1.2 Làm sạch (đếm từ file kết quả)')
    b.code('''lk, ex = js('tables/leakage_before_after.json'), js('runs/train_exclude.json')
pd.DataFrame([
    ['Trùng lặp (ViMedCSS)', lk['after']['dropped_duplicates']],
    ['Clip im lặng, đỉnh biên độ < 0,001', lk['after']['dropped_silent']],
    ['Nhãn dày bất thường (> 16 token/giây), chỉ bỏ khỏi Train', len(ex['ids'])],
], columns=['Bước', 'Số clip loại'])''')
    b.md('Thêm: 10 clip audio rỗng của ViMedCSS train bị loại khi nạp dữ liệu (dòng `Exclude ViMedCSS/train` trong log huấn luyện ở `03_V1`). '
         'Văn bản: NFC, chữ thường, bỏ dấu câu, cùng một luật khi huấn luyện và khi chấm. Nhãn thuật ngữ `cs_terms` do tác giả ViMedCSS cung cấp; '
         'nhóm không gán nhãn thủ công nên không có Kappa.')
    b.md('## 1.3 Mất cân bằng')
    b.code('''im = js('tables/imbalance.json')
w = pd.DataFrame({'% số từ Train': im['train_word_share_pct'], '% số từ DEV': im['dev_word_share_pct']})
ax = w.plot.bar(figsize=(7, 3), color=['#7E9C9B', '#FFB21A'], rot=0); ax.set_ylabel('%'); plt.show()
w''')
    b.code('''print(f"Thuật ngữ trong Train: {im['train_term_distinct']} loại, {im['train_term_occurrences']:,} lượt".replace(',', '.'))
print(f"Chỉ xuất hiện 1 lần: {im['terms_seen_once']}; dưới 5 lần: {im['terms_seen_lt5']}")
print(f"Top-10 thuật ngữ chiếm {im['top10_share_pct']:.1f}% số lượt; riêng '{im['top1_term']}' chiếm {im['top1_share_pct']:.1f}%")''')
    b.md('Thống kê thuật ngữ ở trên tính trên bản chia ViMedCSS theo video (831 loại). Train chính thức có 610 loại (bảng 1.4).')
    b.md('## 1.4 Xử lý mất cân bằng: oversampling clip chứa thuật ngữ hiếm (≤ 5 lần) × 3')
    b.code('''o = pd.read_csv(T / 'term_oversampling.csv').set_index('term_count_bucket')
ax = o[['occ_before', 'occ_after']].rename(columns={'occ_before': 'Trước', 'occ_after': 'Sau'}).plot.bar(figsize=(7, 3), color=['#7E9C9B', '#FFB21A'], rot=0)
ax.set_xlabel('số lần thuật ngữ xuất hiện trong Train'); ax.set_ylabel('số lượt'); plt.show()
rare = o.loc[['1', '2–5']]
print(f"Lượt thuật ngữ hiếm: {rare.occ_before.sum()} → {rare.occ_after.sum()}; tỷ lệ {100 * rare.occ_before.sum() / o.occ_before.sum():.2f}% → {100 * rare.occ_after.sum() / o.occ_after.sum():.2f}%")
o''')
    b.md('## Rò rỉ video trong split gốc của ViMedCSS')
    b.code('''b_, a_ = lk['before'], lk['after']
print(f"Trước: {b_['dev_clips_sharing_video']}/{b_['dev_clips']} clip DEV ({100 * b_['dev_clips_sharing_video'] / b_['dev_clips']:.1f}%) thuộc video có trong Train")
print(f"Sau khi chia theo video: {a_['dev_clips_sharing_video']} clip")''')
    b.build()


def nb02():
    b = Book('02_Baseline.ipynb', '# 02 — Hai baseline chưa huấn luyện (zero-shot)')
    b.md('''Cùng dữ liệu, cùng luật chuẩn hóa, cùng giải mã greedy. Baseline không huấn luyện nên không có lr, optimizer, epoch.
Phần cứng: RTX 4050 Laptop 6 GB; PyTorch 2.6.0, Transformers 4.46.3.''')
    b.code("print((S / 'configs/_base.yaml').read_text(encoding='utf-8'))")
    b.command('01_B0.ipynb', 5, 'Baseline 1 trên toàn bộ DEV')
    b.code("show_log('Baseline1_dev_eval.txt')")
    b.code("show_log('Baseline2_dev_eval.txt')")
    b.code(DEV_TABLE.replace("'V1': dev_row('V1_dev'),\n                    'V2': dev_row('V2_dev'), 'V-Final': dev_row('VFinal_dev_beam5')", ''))
    b.md('## So với số công bố (paper ViMedCSS), WER %')
    b.code('''# ViMedCSS TEST (1.613 clip) và HARD (658 clip); B2, B3 là PhoWhisper-large, Whisper-large-v3 chưa huấn luyện, chạy thêm để kiểm tra cách chấm
g = pd.read_csv(T / 'same_metric_official.csv').query("system in ['B0', 'B2', 'B3']")
p = pd.read_csv(T / 'same_metric_paper.csv').set_index(['split', 'system'])['wer']
g = g.assign(paper=[p.get((r.split, r.paper_row)) for r in g.itertuples()])[['split', 'paper_row', 'paper', 'wer']]
g = g.assign(lech=g.paper - g.wer).rename(columns={'paper_row': 'mô hình', 'wer': 'nhóm', 'lech': 'lệch'}).reset_index(drop=True)
g.loc[len(g)] = ['TEST', 'Whisper-Small zero-shot (Baseline 2)', p[('TEST', 'Whisper-Small zero-shot')], float('nan'), float('nan')]
g''')
    b.md('''**Gap analysis.** Cùng mô hình, cùng tập, không huấn luyện mà số của nhóm thấp hơn paper 5–10 điểm, nên chênh lệch đến từ
cách chấm (transcript tham chiếu, chuẩn hóa văn bản, scorer), không phải từ siêu tham số. Paper không công bố luật chuẩn hóa.
Baseline 2 chưa được giải mã trên TEST/HARD chính thức, nên chưa có dòng so trực tiếp với paper.''')
    b.build()


def nb03():
    b = Book('03_V1.ipynb', '# 03 — V1: PhoWhisper-small + LoRA, trộn 4 nguồn')
    b.md('''**Kiến trúc.** Log-Mel 80 × 3.000 → Conv1D × 2 → Encoder 12 lớp → Decoder 12 lớp → token. LoRA gắn vào q, k, v, out_proj,
fc1, fc2 của encoder và decoder: `y = W·x + (α/r)·B·A·x`, W đóng băng, r = 32, α = 64, dropout 0,05.
**So với baseline:** thêm LoRA, huấn luyện trên 4 nguồn, nhãn chuẩn hóa cùng luật chấm.''')
    b.code("print((S / 'configs/V1.yaml').read_text(encoding='utf-8'))")
    b.command('03_V1.ipynb', 5, 'huấn luyện V1')
    b.code("show_log('V1_train.txt')")
    b.md('## Đường cong loss và WER (từ `trainer_state.json`)')
    b.code('''h = history('V1')
tr = pd.DataFrame([x for x in h if 'loss' in x and 'step' in x]); ev = pd.DataFrame([x for x in h if 'eval_wer' in x])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
ax[0].plot(tr.step, tr.loss, lw=1, color='#0F4E57'); ax[0].set_title('Train loss')
ax[1].plot(ev.step, ev.eval_loss, marker='o', ms=3, color='#BC1F45'); ax[1].set_title('Eval loss (DEV-subset 600 clip)')
ax[2].plot(ev.step, 100 * ev.eval_wer, marker='o', ms=3, color='#CF8700'); ax[2].set_title('Eval WER % (DEV-subset)')
for a in ax: a.set_xlabel('bước')
plt.tight_layout(); plt.show()
pk = tr.loc[tr.loss.idxmax()]
print(f"Loss đỉnh {pk.loss:.2f} ở bước {int(pk.step)} (grad_norm {pk.grad_norm:.0f}); eval loss cao nhất {ev.eval_loss.max():.2f} ở bước {int(ev.loc[ev.eval_loss.idxmax(), 'step'])}")''')
    b.md('## Tín hiệu cải thiện so với 2 baseline (DEV chính thức)')
    b.code(DEV_TABLE.replace(",\n                    'V2': dev_row('V2_dev'), 'V-Final': dev_row('VFinal_dev_beam5')", ''))
    b.code('''d = DEV.loc['V1'] - DEV.loc['Baseline 1']
print('V1 so với Baseline 1 (điểm %):'); print(d.round(2).to_string())''')
    b.md('## Phân tích lỗi của V1 (toàn bộ DEV, cùng phương pháp phân loại với báo cáo)')
    b.code('''import collections, jiwer
pred = pd.read_json(S / 'predictions/V1_dev.jsonl', lines=True)
pred = pred[pred.ref != ''].reset_index(drop=True)
subs, recs = collections.Counter(), []
for i, r in pred.iterrows():
    o = jiwer.process_words(r.ref, r.hyp); rw, hw = r.ref.split(), r.hyp.split(); al = o.alignments[0]
    for ch in al:
        if ch.type == 'substitute':
            for a, b in zip(range(ch.ref_start_idx, ch.ref_end_idx), range(ch.hyp_start_idx, ch.hyp_end_idx)):
                subs[(rw[a], hw[b])] += 1
    edge = sum(max(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx) for c in (al[0], al[-1]) if c.type in ('delete', 'insert')) if al else 0
    recs.append(dict(source=r.source, n=max(1, len(rw)), S=o.substitutions, D=o.deletions, I=o.insertions, edge=edge))
err = pd.DataFrame(recs); err['errors'] = err.S + err.D + err.I
cat = pd.Series('thay thế rải rác', index=err.index)
cat[err.D > err.S + err.I] = 'xóa là chính (mất đoạn)'
cat[err.I > err.S + err.D] = 'chèn là chính (thừa/lặp)'
cat[(err.edge >= 3) & (err.edge >= .4 * err.errors)] = 'lệch biên đầu/cuối clip'
cat[err.errors == 0] = 'đúng hoàn toàn'
err['category'] = cat
total = err.errors.sum()
by_cat = err[err.errors > 0].groupby('category').agg(clip=('n', 'size'), loi=('errors', 'sum')).assign(pct=lambda x: 100 * x.loi / total).sort_values('loi', ascending=False)
print(f"{total:,} lỗi từ trên {len(err):,} clip; {int((err.errors == 0).sum()):,} clip đúng hoàn toàn".replace(',', '.'))
by_cat''')
    b.code('''fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].barh(by_cat.index[::-1], by_cat.pct[::-1], color='#BC1F45'); ax[0].set_xlabel('% số lỗi'); ax[0].set_title('Loại lỗi')
src = err.groupby('source').errors.sum() / total * 100
ax[1].bar(src.index, src.values, color='#0F4E57'); ax[1].set_ylabel('% số lỗi'); ax[1].set_title('Lỗi theo nguồn')
plt.tight_layout(); plt.show()''')
    b.md('### Cặp từ hay nhầm (thay cho Confusion Matrix: ASR không có lớp cố định)')
    b.code('''top = subs.most_common(20)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh([f'{a} → {b}' for (a, b), _ in top][::-1], [c for _, c in top][::-1], color='#4C78A8'); ax.set_xlabel('số lần'); plt.show()''')
    b.code('''import numpy as np
refs = [w for w, _ in collections.Counter(a for (a, _), c in subs.items() for _ in range(c)).most_common(10)]
hyps = [w for w, _ in collections.Counter(b for (a, b), c in subs.items() if a in refs for _ in range(c)).most_common(10)]
M = np.array([[subs.get((a, b), 0) for b in hyps] for a in refs])
fig, ax = plt.subplots(figsize=(7, 5.5)); im = ax.imshow(M, cmap='Blues')
ax.set_xticks(range(len(hyps)), hyps, rotation=45, ha='right'); ax.set_yticks(range(len(refs)), refs)
for i in range(len(refs)):
    for j in range(len(hyps)):
        if M[i, j]: ax.text(j, i, M[i, j], ha='center', va='center', fontsize=8, color='white' if M[i, j] > M.max() / 2 else 'black')
ax.set_xlabel('mô hình nghe thành'); ax.set_ylabel('từ đúng'); ax.set_title('Ma trận nhầm lẫn giữa 10 từ hay bị thay nhất'); plt.colorbar(im); plt.show()''')
    b.md('### Thuật ngữ y khoa: đã gặp và chưa gặp khi huấn luyện')
    b.code('''train_terms = js('runs/train_terms_official.json')
occ, word, sub = collections.Counter(), collections.Counter(), collections.Counter()
for r in pred.itertuples():
    w, seen = r.hyp.split(), collections.Counter()
    for t in r.terms:
        tw = t.split(); seen[t] += 1
        k = sum(w[i:i + len(tw)] == tw for i in range(len(w) - len(tw) + 1))
        occ[t] += 1; word[t] += seen[t] <= k; sub[t] += t in r.hyp
tm = pd.DataFrame({'lượt': occ, 'đúng nguyên từ': word, 'đúng chuỗi con': sub}).fillna(0)
n = tm['lượt'].sum()
print(f"Recall thuật ngữ: khớp chuỗi con {100 * tm['đúng chuỗi con'].sum() / n:.2f}% (bằng số trong metrics.json), "
      f"khớp nguyên từ {100 * tm['đúng nguyên từ'].sum() / n:.2f}% (chặt hơn: 'vaccin' trong 'vaccine' không được tính)")
unseen = [t for t in tm.index if t not in train_terms]
print(f"Thuật ngữ trong DEV chưa từng có trong Train chính thức: {len(unseen)} loại")
tm.assign(sai=tm['lượt'] - tm['đúng chuỗi con']).sort_values('sai', ascending=False).head(10)''')
    b.md('### Ví dụ sai nhiều nhất (câu 6–28 từ)')
    b.code('''mix = pred.join(err[['S', 'D', 'I', 'errors', 'category']]).assign(wer=lambda x: x.errors / x.ref.str.split().str.len())
ex = mix[mix.ref.str.split().str.len().between(6, 28)].sort_values('wer', ascending=False).groupby('source').head(2)
pd.set_option('display.max_colwidth', 120); ex[['source', 'category', 'wer', 'ref', 'hyp']]''')
    b.build()


def nb04():
    b = Book('04_V2.ipynb', '# 04 — V2: train tiếp V1 với lr 1e-4')
    b.md('''**Lý do từ phân tích lỗi V1:** lr 1e-3 làm huấn luyện phân kỳ (loss và eval loss vọt lên quanh bước 700–900, xem `03_V1`).
V2 khởi tạo từ adapter tốt nhất của V1 và train thêm 2 epoch với lr thấp hơn 10 lần; mọi thứ khác giữ nguyên.''')
    b.code("print((S / 'configs/V2.yaml').read_text(encoding='utf-8'))")
    b.command('08_V2.ipynb', 5, 'huấn luyện V2 (chạy nền, log lưu ra file)')
    b.code("show_log('V2_train.txt')")
    b.code("show_log('V2_dev_eval.txt')")
    b.md('## V1 và V2 trong lúc train (cùng DEV-subset 600 clip)')
    b.code('''fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
for name, col, lab in [('V1', '#BC1F45', 'V1 (lr 1e-3)'), ('V2', '#0F4E57', 'V2 (tiếp từ V1, lr 1e-4)')]:
    h = history(name); tr = pd.DataFrame([x for x in h if 'loss' in x and 'step' in x]); ev = pd.DataFrame([x for x in h if 'eval_wer' in x])
    ax[0].plot(tr.step, tr.loss, lw=1, color=col, label=lab); ax[1].plot(ev.step, (100 * ev.eval_wer).clip(upper=40), marker='o', ms=3, color=col, label=lab)
ax[0].set_title('Train loss'); ax[1].set_title('Eval WER % (DEV-subset, cắt ở 40)')
for a in ax: a.set_xlabel('bước (mỗi run đếm từ 0)'); a.legend(frameon=False)
plt.tight_layout(); plt.show()''')
    b.md('## Kết quả trên toàn bộ DEV chính thức')
    b.code(DEV_TABLE.replace("'Baseline 1': dev_row('Baseline1_dev'), 'Baseline 2': dev_row('Baseline2_dev'), ", '').replace(", 'V-Final': dev_row('VFinal_dev_beam5')", ''))
    b.code("(DEV.loc['V2'] - DEV.loc['V1']).round(2).rename('V2 − V1 (điểm %)')")
    b.md('**Nhận xét:** V2 không cải thiện WER tổng; train tiếp với lr thấp không sửa được hậu quả của đợt phân kỳ. '
         'Vì vậy V-Final huấn luyện lại từ đầu với lr 1e-4 và xử lý thêm các nguyên nhân lỗi khác.')
    b.build()


def nb05():
    b = Book('05_V_Final.ipynb', '# 05 — V-Final: Whisper-large-v3 + LoRA + SpecAugment + oversampling thuật ngữ hiếm, beam 5')
    b.md('''| Lỗi thấy được ở V1, V2 | Thay đổi ở V-Final |
|---|---|
| lr 1e-3 phân kỳ; train tiếp (V2) không sửa được | huấn luyện lại từ đầu, lr 1e-4 |
| quá khớp về cuối huấn luyện | SpecAugment: che 5% thời gian + 5% tần số |
| thuật ngữ hiếm, chưa gặp bị sai nhiều | oversampling clip chứa thuật ngữ xuất hiện ≤ 5 lần, lặp × 3 |
| backbone nhỏ | Whisper-large-v3, 1,55 tỉ tham số, đóng băng |
| lỗi giải mã | beam 5, chọn trên DEV trước khi xem TEST |

Huấn luyện trên Train chính thức của 4 nguồn để chấm được trên đúng TEST/HARD của paper. Colab A100 80 GB, bf16.''')
    b.code("print((S / 'configs/VFinal.yaml').read_text(encoding='utf-8'))")
    for idx, label in [(4, 'huấn luyện trên Colab'), (6, 'DEV greedy và beam 5'), (7, 'chốt mô hình trước khi giải mã TEST'), (9, 'TEST và HARD, mỗi tập một lần')]:
        src = nbformat.read(ASR / 'colab_budget.ipynb', as_version=4).cells[idx].source.strip()
        b.md(f'**Lệnh đã chạy — {label}** (chép từ `ASR/colab_budget.ipynb`, ô {idx}; chỉ để xem)\n\n```python\n{src}\n```')
    b.md('## Nhật ký lần chạy trên Colab')
    b.code('''recs = [json.loads(l) for l in (S / 'runs/VFinal_runs.jsonl').read_text(encoding='utf-8').splitlines()]
pd.DataFrame([{'thời điểm': r['time'], 'việc': r['kind'], 'thiết bị': r['device'], 'số clip train': r.get('n_train', ''), 'torch': r['versions']['torch']} for r in recs])''')
    b.code('''fz = js('runs/VFinal_freeze.json')
print(json.dumps(fz, ensure_ascii=False, indent=2)[:1500])''')
    b.md('Log loss từng bước của V-Final nằm trong checkpoint trên Google Drive, chưa tải về máy, nên chưa có đường cong huấn luyện ở đây.')
    b.md('## Chọn cách giải mã trên DEV, rồi TEST/HARD')
    b.code('''rows = {}
for k, lab in [('dev', 'DEV greedy'), ('dev_beam5', 'DEV beam 5'), ('test_beam5', 'TEST cả 4 nguồn, beam 5'), ('hard_beam5', 'ViMedCSS HARD, beam 5')]:
    m = js(f'metrics/VFinal_{k}.json'); o = m['overall']
    rows[lab] = {'WER %': 100 * o['wer'], 'CER %': 100 * o['cer'], 'Recall thuật ngữ %': 100 * o.get('cs_term_recall', float('nan')), 'số clip': o['n_utts'], 'RTF (A100)': m.get('rtf_decode_only')}
t = pd.DataFrame(rows).T.astype({'số clip': int})
t['RTF (A100)'] = t['RTF (A100)'].map('{:.3f}'.format)
t''')
    b.md('## So với mô hình chưa huấn luyện và với paper (ViMedCSS TEST/HARD)')
    b.code('''s = pd.read_csv(T / 'same_metric_official.csv'); p = pd.read_csv(T / 'same_metric_paper.csv')
lab = {'B0': 'PhoWhisper-small', 'B2': 'PhoWhisper-large', 'B3': 'Whisper-large-v3', 'O3W': 'V-Final'}
w = s[s.system.isin(lab)].assign(system=lambda x: x.system.map(lab)).pivot_table(index='system', columns='split', values='wer').reindex(list(lab.values()))
ax = w.plot.bar(figsize=(7.5, 3.2), rot=0, color=['#0F4E57', '#FFB21A'])
ax.set_ylabel('WER %'); ax.set_title('ViMedCSS, cùng tập, cùng cách chấm'); plt.show()
w''')
    b.code('''pd.read_csv(T / 'official_test_by_source.csv').set_index('system').rename(index={'B0': 'PhoWhisper-small', 'B2': 'PhoWhisper-large', 'B3': 'Whisper-large-v3', 'O3W': 'V-Final'}).drop(columns='description')''')
    b.md('Số công bố trong paper ViMedCSS (WER %, CS-WER %). Hệ tốt nhất của paper là PhoWhisper-Small + AG: TEST 23,67, HARD 33,73. '
         'Chênh lệch cách chấm 5–10 điểm (notebook `02_Baseline`) nên đây chưa phải xếp hạng.')
    b.code("p.set_index(['split', 'system'])")
    b.md('## Ví dụ đầu ra trên TEST (mẫu ngẫu nhiên, seed 13, clip có thuật ngữ)')
    b.code('''pt = pd.read_json(S / 'predictions/VFinal_test.jsonl', lines=True)
pt = pt[pt.terms.str.len() > 0]
def hit(r): return all(t in r.hyp for t in r.terms)
sm = pt.sample(10, random_state=13).assign(thuat_ngu=lambda x: x.terms.str.join('; '), dung_thuat_ngu=lambda x: x.apply(hit, axis=1))
pd.set_option('display.max_colwidth', 110); sm[['thuat_ngu', 'dung_thuat_ngu', 'ref', 'hyp']]''')
    b.build()


def nb06():
    b = Book('06_Thanh_phan_Kha_thi.ipynb', '# 06 — Thí nghiệm thành phần và tính khả thi')
    b.md('''## Thí nghiệm thành phần (ViMedCSS chia theo video)
Các run này đo tác dụng riêng của từng thành phần trước khi gộp vào V-Final. Chúng dùng ViMedCSS chia lại theo video (80/10/10,
seed 13, không trùng video giữa các tập), nên số **không so trực tiếp** với DEV chính thức ở các notebook trước.''')
    b.code("show_log('thanh_phan_chia_theo_video_train.txt')")
    b.code("show_log('thanh_phan_specaugment_train.txt')")
    b.code("show_log('thanh_phan_medium_train.txt')")
    b.code('''c = pd.read_csv(T / 'eval_curves.csv')
lab = {'V3': 'chia theo video, lr 3e-4', 'V4': '+ SpecAugment', 'V5': 'PhoWhisper-medium + SpecAugment'}
fig, ax = plt.subplots(figsize=(7, 3.2))
for k, l in lab.items():
    d = c[c.run == k]; ax.plot(d.step, d.eval_wer, marker='o', ms=3, label=l)
ax.set_xlabel('bước'); ax.set_ylabel('WER % DEV-subset'); ax.legend(frameon=False); plt.show()''')
    b.code('''d = pd.read_csv(T / 'ablation_dev_v3.csv').set_index('run')
lab = {'B0V': 'Baseline 1', 'V3': 'chia theo video', 'V3B': 'chia theo video, beam 5', 'V4': '+ SpecAugment', 'V4B': '+ SpecAugment, beam 5', 'V5': 'medium', 'V5B': 'medium, beam 5'}
d = d.loc[[k for k in lab if k in d.index], ['decode', 'wer', 'term_recall', 'wer_ViMedCSS', 'wer_VietMed', 'rtf']].rename(index=lab)
d''')
    b.code('''w = d.wer
print(f"SpecAugment: {w['+ SpecAugment'] - w['chia theo video']:+.2f} điểm (greedy)")
print(f"Beam 5: {w['+ SpecAugment, beam 5'] - w['+ SpecAugment']:+.2f} điểm (cùng mô hình SpecAugment)")
print(f"Backbone medium (beam 5): {w['medium, beam 5'] - w['+ SpecAugment, beam 5']:+.2f} điểm")''')
    b.md('## Tính khả thi: độ chính xác, tốc độ, kích thước')
    b.code('''l = pd.read_csv(T / 'latency_size.csv').set_index('system')
l = l.rename(index={'V4 (small)': 'small + SpecAugment, greedy', 'V4B (small)': 'small + SpecAugment, beam 5', 'V5 (medium)': 'medium, greedy', 'V5B (medium)': 'medium, beam 5'})
vf = js('metrics/VFinal_test_beam5.json')
l.loc['V-Final (A100, beam 5)'] = {'params_M': 1550, 'adapter_MB': js('runs/VFinal_adapter.json')['adapter_MB'], 'dev_wer': 100 * js('metrics/VFinal_dev_beam5.json')['overall']['wer'], 'rtf': vf['rtf_decode_only']}
l = l[['params_M', 'adapter_MB', 'dev_wer', 'rtf', 'eval_gpu_gb', 'train_gpu_gb']].astype({'params_M': int})
l['rtf'] = l['rtf'].map('{:.3f}'.format)
l''')
    b.md('''- RTF = thời gian giải mã / thời lượng audio. Các dòng small/medium đo trên RTX 4050 6 GB, V-Final đo trên A100 (batch 16), nên không so trực tiếp.
- `dev_wer` của small/medium là DEV chia theo video; của V-Final là DEV chính thức.
- Chưa đo VRAM và tốc độ của V-Final trên GPU 6 GB.

**Kết luận: CÓ khả thi, có điều kiện.** Dùng làm bản nháp có người soát (còn sót khoảng 2/10 đến 3/10 thuật ngữ trên TEST);
máy 6 GB dùng bản medium, V-Final khi có GPU từ 8 GB và đã đo lại.''')
    b.build()


if __name__ == '__main__':
    collect()
    print('saved/:', sum(f.stat().st_size for f in SAVED.rglob('*') if f.is_file()) / 2**20, 'MB', flush=True)
    for f in (nb00, nb01, nb02, nb03, nb04, nb05, nb06):
        f()
