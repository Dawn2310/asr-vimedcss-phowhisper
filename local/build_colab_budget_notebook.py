"""Regenerate ASR/colab_budget.ipynb: the one-training-run version of colab_official.ipynb (~45 Colab units)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# Colab A100 (bản tiết kiệm) — 1 lần train trên split CHÍNH THỨC, so trực tiếp với paper

**Chạy: Thời gian chạy → Chạy tất cả.** Tổng ~3,5–4 giờ A100 (~35 đơn vị). Giữ tab mở.

---

## Giới thiệu

**Mục tiêu:** có số trên đúng các tập test mà paper dùng:
ViMedCSS TEST + Hard 658 clip (paper ViMedCSS 2026), VIVOS test (PhoWhisper), VietMed test (VietMed).
Vì vậy chỉ train trên phần Train chính thức (V3+ đã thấy clip TEST chính thức nên không dùng được ở đây).

| Bước | Thí nghiệm | Ghi chú |
|---|---|---|
| 0 | B0, B2, B3 zero-shot trên DEV | tái lập số paper (PhoWhisper-small, PhoWhisper-large, Whisper-large-v3) |
| 1 | O1 PhoWhisper-large + LoRA + SpecAugment | |
| 2 | O2 Whisper-large-v3 + LoRA + SpecAugment | backbone đa ngôn ngữ, viết thuật ngữ Anh tốt hơn |
| 3 | O3P hoặc O3W = bản thắng ở bước 1–2 + oversampling thuật ngữ hiếm | chọn tự động theo DEV |
| 4 | beam 5 trên DEV, chọn hệ cuối, **chốt (freeze)** | chỉ dùng DEV |
| 5 | TEST + HARD một lần cho hệ cuối và các zero-shot | |

**Chuẩn bị trên Drive** (`MyDrive/DPL/ASR`): cả thư mục `ASR/` gồm code, `experiments/`, `requirements.txt` và `data/` với
`VIVOS/`, `FLEURS/`, `VietMed/`, `ViMedCSS/`, `registry.json`, `v3_train_exclude.json` (~4,1 GB).
**Không cần** `data/ViMedCSS_vsplit/`, `local/`, `.venv`.

Runtime → Change runtime type → **A100 GPU**. Mỗi lần train ước tính 2–4 giờ (xem ETA ở log epoch đầu).
Checkpoint ghi lên Drive mỗi epoch; mất kết nối thì chạy lại cell setup và cell train đó, nó tự chạy tiếp.
''')
code('''from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/DPL/ASR'  # sửa nếu đặt chỗ khác
%cd {PROJECT}
!pip install -q -r requirements.txt
import os, json, subprocess
os.environ['HF_HOME'] = '/content/hf_cache'          # model tải về ổ local của Colab, không chiếm Drive
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout
print(gpu)
assert 'A100' in gpu, 'Cần runtime A100 (config bs 32, bf16 tính cho A100)'
''')
code('''# Copy data từ Drive sang ổ local của Colab (đọc nhanh hơn nhiều khi train)
!mkdir -p /content/data
for s in ['VIVOS', 'FLEURS', 'VietMed', 'ViMedCSS']:
    !rsync -a --info=progress2 {PROJECT}/data/{s} /content/data/
!cp {PROJECT}/data/registry.json {PROJECT}/data/v3_train_exclude.json /content/data/
DATA = '/content/data'

import sys
os.environ['TQDM_MININTERVAL'] = '30'   # progress bars refresh every 30 s instead of flooding the cell

def run(*args):
    """Stream the child's output into the cell (os.system output never reaches a Colab cell)."""
    cmd = ['python', '-u', *map(str, args), '--data-dir', DATA]
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for chunk in iter(lambda: p.stdout.read(256), ''):
        sys.stdout.write(chunk); sys.stdout.flush()
    assert p.wait() == 0, f'lỗi (exit {p.returncode}), xem log phía trên'

def dev_metrics(exp, name='dev'):
    p = f'{PROJECT}/experiments/{exp}/eval/{name}/metrics.json'
    return json.load(open(p)) if os.path.exists(p) else None

def evaluate_once(exp, split='dev', beams=None):
    """Decode once; skip if the output already exists (TEST/HARD are never re-run)."""
    name = split + (f'_beam{beams}' if beams else '')
    if dev_metrics(exp, name):
        print('Đã có, bỏ qua:', exp, name); return
    run('scripts/evaluate.py', exp, '--split', split, *(['--allow-test'] if split in ('test', 'hard') else []),
        *(['--num-beams', beams] if beams else []))
''')
md('''## 1. Train O3W = Whisper-large-v3 + LoRA + SpecAugment + oversampling thuật ngữ hiếm (~1,5–2,5 giờ)
Ngân sách chỉ đủ một lần train, nên chọn backbone viết thuật ngữ tiếng Anh tốt nhất trong paper (Whisper-large-v3).
''')
code('''EXP = 'O3W_whisper_large_v3_lora_official_termos'
run('scripts/train.py', EXP)
''')
md('''## 2. DEV: greedy và beam 5, chọn cách giải mã tốt hơn
''')
code('''evaluate_once(EXP)
evaluate_once(EXP, beams=5)
import pandas as pd
rows = []
for name in ['dev', 'dev_beam5']:
    m = dev_metrics(EXP, name)
    rows.append(dict(exp=EXP, decode=name, wer=100*m['overall']['wer'], vimedcss=100*m['by_source']['ViMedCSS']['wer'],
                     term_recall=100*m['overall']['cs_term_recall'], truncated=m['overall']['n_truncated']))
table = pd.DataFrame(rows).sort_values('wer')
display(table.round(2))
FINAL = table.iloc[0].to_dict()
print('Hệ cuối:', FINAL)
''')
code('''# Freeze: ghi lại hệ cuối (config, sha256 adapter, cách giải mã, DEV) TRƯỚC khi giải mã TEST. Không ghi đè.
import hashlib, datetime, yaml
exp = FINAL['exp']
out = f'{PROJECT}/reports/final_freeze_official'
assert not os.path.exists(f'{out}/freeze.json'), 'đã chốt rồi, không chốt lại'
assert not any(os.path.exists(f'{PROJECT}/experiments/{exp}/eval/{s}') for s in ('test', 'test_beam5', 'hard', 'hard_beam5')), 'đã có TEST'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
record = dict(frozen_at=datetime.datetime.now().isoformat(timespec='seconds'), final_system=exp,
              num_beams=5 if FINAL['decode'] == 'dev_beam5' else 1,
              config=yaml.safe_load(open(f'{PROJECT}/experiments/{exp}/config.yaml')),
              adapter_sha256=sha(f'{PROJECT}/experiments/{exp}/final/adapter_model.safetensors'),
              selection='lowest official-DEV WER of O3W greedy vs beam 5; TEST not decoded yet',
              dev_table=table.round(4).to_dict('records'))
os.makedirs(out, exist_ok=True)
json.dump(record, open(f'{out}/freeze.json', 'w'), ensure_ascii=False, indent=1)
print('Đã chốt:', exp, 'beam', record['num_beams'])
''')
md('''## 5. TEST + HARD chính thức, mỗi thứ đúng một lần
Zero-shot B0/B2/B3 cũng chạy để so cách chấm điểm của nhóm với số paper.
''')
code('''freeze = json.load(open(f'{PROJECT}/reports/final_freeze_official/freeze.json'))
beams = 5 if freeze['num_beams'] == 5 else None
for split in ['hard', 'test']:
    evaluate_once(freeze['final_system'], split, beams)
for exp in ['B0_phowhisper_small_zeroshot', 'B2_phowhisper_large_zeroshot', 'B3_whisper_large_v3_zeroshot']:
    for split in ['hard', 'test']:
        evaluate_once(exp, split)
''')
code('''# Bảng so với paper (WER %, cùng tập TEST/Hard chính thức). CS-WER tính sau ở máy local.
import pandas as pd
freeze = json.load(open(f'{PROJECT}/reports/final_freeze_official/freeze.json'))
PAPER ={'ViMedCSS TEST': {'PhoWhisper-small zero-shot': 36.31, 'PhoWhisper-large zero-shot': 31.24,
                           'Whisper-large-v3 zero-shot': 34.47, 'VietASR zero-shot': 27.56,
                           'small + LoRA': 27.13, 'small + Attention Guide (best)': 23.67},
         'ViMedCSS HARD': {'PhoWhisper-small zero-shot': 46.27, 'PhoWhisper-large zero-shot': 37.37,
                           'Whisper-large-v3 zero-shot': 39.35, 'small + Attention Guide (best)': 33.73},
         'VIVOS test': {'PhoWhisper-small': 6.33, 'PhoWhisper-large': 4.67},
         'VietMed test': {'XLSR-53-Viet (VietMed paper)': 29.6}}
ours = {}
for exp, label in [('B0_phowhisper_small_zeroshot', 'B0 small zero-shot'), ('B2_phowhisper_large_zeroshot', 'B2 large zero-shot'),
                   ('B3_whisper_large_v3_zeroshot', 'B3 large-v3 zero-shot'), (freeze['final_system'], 'HỆ CUỐI')]:
    sfx = f'_beam{freeze["num_beams"]}' if label == 'HỆ CUỐI' and freeze['num_beams'] == 5 else ''
    t, h = dev_metrics(exp, 'test' + sfx), dev_metrics(exp, 'hard' + sfx)
    if t:
        ours[label] = {'ViMedCSS TEST': 100*t['by_source']['ViMedCSS']['wer'], 'VIVOS test': 100*t['by_source']['VIVOS']['wer'],
                       'VietMed test': 100*t['by_source']['VietMed']['wer'], 'FLEURS test': 100*t['by_source']['FLEURS']['wer'],
                       'ViMedCSS HARD': 100*h['overall']['wer'] if h else None,
                       'term recall TEST': 100*t['by_source']['ViMedCSS']['cs_term_recall']}
display(pd.DataFrame(ours).round(2))
for k, v in PAPER.items():
    print(k, v)
''')
md('''## Mang kết quả về máy local
Tải về các thư mục `experiments/B0_*`, `B2_*`, `B3_*`, `O1_*`, `O2_*`, `O3*_*` (chỉ cần `eval/`, `final/`, `runs.jsonl`;
bỏ `checkpoints/` cho nhẹ) và `reports/final_freeze_official/`, chép đè vào `ASR/` trên máy. Sau đó báo để cập nhật báo cáo.
''')
for i, cell in enumerate(cells):
    cell['id'] = f'colab-budget-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'accelerator': 'GPU', 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'colab_budget.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
