"""Regenerate ASR/colab_paper.ipynb: the paper experiments (A leakage study, B data/backbone ablation on the official
split, C term-prompt training + biased decoding) on a Colab A100. Every cell is idempotent: finished trainings and
decodes are skipped, so after a disconnect re-run the setup cells and then the cell that was running."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# Colab A100 — thí nghiệm cho paper (A + B + C)

| Phần | Câu hỏi | Thí nghiệm | A100 ước tính |
|---|---|---|---|
| **A1** | Rò rỉ video làm WER đẹp lên bao nhiêu? | PhoWhisper-small, Train sạch và Train rò rỉ (bằng nhau về số clip), 3 seed | ~3 giờ |
| **B1** | Kết quả tốt là nhờ thêm dữ liệu hay nhờ mô hình? | small + LoRA: chỉ ViMedCSS (giống paper) → +VietMed → đủ 4 nguồn; O2 (O3W bỏ oversampling) | ~4,5 giờ |
| **C1** | Gợi ý thuật ngữ có cứu được thuật ngữ chưa thấy (HARD)? | O3W + huấn luyện có prompt thuật ngữ; giải mã có danh sách thuật ngữ | ~5,5 giờ |
| A2 | Rò rỉ có còn khi mô hình lớn? | Cặp sạch/rò rỉ cho medium và large-v3 | ~6,5 giờ |
| B2 | Backbone, đối chứng oversampling, VIVOS | large-v3 chỉ ViMedCSS; O2R (lặp clip ngẫu nhiên, cùng số bước với O3W); O3P (PhoWhisper-large) | ~7 giờ |

Thứ tự ưu tiên: **A1 → B1 → A2 → B2**, C1 cuối cùng (lý do ở cuối `reports/paper/PAPER_PLAN.md`). Tổng ~26,5 giờ A100 (~265 đơn vị).
Thời gian là ước lượng từ lần chạy O3W (2 epoch large-v3 ≈ 2,4 giờ) cho chạy tuần tự, chưa tính phần tăng tốc bên dưới. Sau A1 nên xem kết quả trước khi chạy tiếp
(xem `reports/paper/PAPER_PLAN.md`).

**Tận dụng 80 GB:** các job chạy song song trên cùng GPU và ghi log vào `/content/logs/<tên>.log`; mô hình lớn tắt gradient
checkpointing. **Không tăng batch:** batch train 32 là một phần công thức để so với O3W, còn batch giải mã thay đổi giới hạn độ dài
của decode guard. Mỗi ô in thời gian thực tế khi xong.

**Cách chạy:** mỗi lần mở phiên mới chạy 3 ô đầu (setup, dữ liệu, hàm dùng chung), sau đó chạy ô của phần cần làm.
Ô nào cũng **bỏ qua phần đã xong** (mô hình đã có `final/`, giải mã đã có `metrics.json`), nên mất kết nối thì chạy lại đúng ô đó.
Train bị ngắt sẽ tự chạy tiếp từ checkpoint; checkpoint được xóa sau khi có `final/` để tiết kiệm Drive.

**Quy tắc chống "chọn trên TEST":** mọi công thức được cố định trong `experiments/<ID>/config.yaml` trước khi chạy;
TEST/HARD/probe chỉ giải mã một lần cho mỗi mô hình và mọi kết quả đều được báo cáo, kể cả kết quả xấu.
''')
code('''from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/DPL/ASR'  # sửa nếu đặt chỗ khác
%cd {PROJECT}
!pip install -q -r requirements.txt
import os, json, subprocess, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
os.environ['HF_HOME'] = '/content/hf_cache'          # model tải về ổ local của Colab, không chiếm Drive
os.environ['TQDM_MININTERVAL'] = '30'                # thanh tiến độ 30 giây cập nhật một lần
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout
print(gpu)
print(os.cpu_count(), 'CPU,', round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30), 'GB RAM')
assert 'A100' in gpu, 'Cần runtime A100 (config bs 32, bf16 tính cho A100)'
''')
code('''# Dữ liệu: copy 4 nguồn từ Drive sang ổ local; ViMedCSS_vsplit và ViMedCSS_leak lấy từ Drive nếu đã có,
# chưa có thì dựng lại từ ViMedCSS chính thức (tất định theo sha256) và kiểm tra trùng id với bản dựng ở máy local.
DATA = '/content/data'
os.makedirs(DATA, exist_ok=True)
for s in ['VIVOS', 'FLEURS', 'VietMed', 'ViMedCSS']:
    subprocess.run(['rsync', '-a', f'{PROJECT}/data/{s}', f'{DATA}/'], check=True)
for f in ['registry.json', 'v3_train_exclude.json']:
    shutil.copy(f'{PROJECT}/data/{f}', DATA)

def sh(*cmd, log=None):
    if log:                                  # job song song: ghi ra file, không in lẫn vào ô
        with open(log, 'a') as f:
            r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
        assert r.returncode == 0, f'lỗi (exit {r.returncode}), xem {log}'
        return
    print('$', ' '.join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for chunk in iter(lambda: p.stdout.read(256), ''):
        sys.stdout.write(chunk); sys.stdout.flush()
    assert p.wait() == 0, f'lỗi (exit {p.returncode}), xem log phía trên'

derived = ['ViMedCSS_vsplit', 'ViMedCSS_leak']
if all(os.path.exists(f'{PROJECT}/data/{d}') for d in derived):
    for d in derived:
        subprocess.run(['rsync', '-a', f'{PROJECT}/data/{d}', f'{DATA}/'], check=True)
    shutil.copy(f'{PROJECT}/data/leak_train_exclude.json', DATA)
else:
    sh('python', '-u', 'scripts/build_v3_data.py', '--data-dir', DATA)
    sh('python', '-u', 'scripts/build_leak_data.py', '--data-dir', DATA)
    for d in derived:                        # lưu lên Drive cho các phiên sau (~2,2 GB)
        subprocess.run(['rsync', '-a', f'{DATA}/{d}', f'{PROJECT}/data/'], check=True)
    shutil.copy(f'{DATA}/leak_train_exclude.json', f'{PROJECT}/data/')
mine = json.load(open(f'{DATA}/ViMedCSS_leak/split_report.json'))['id_sha256']
ref = json.load(open(f'{PROJECT}/reports/paper/leak_split_report.json'))['id_sha256']
assert mine == ref, f'dữ liệu khác bản local: {mine} vs {ref}'
print('Dữ liệu khớp bản local (id hash):', list(mine))
''')
code('''# Hàm dùng chung
def run(*args, log=None):
    """Stream the child's output into the cell (os.system output never reaches a Colab cell), or into `log`."""
    sh('python', '-u', *args, '--data-dir', DATA, log=log)

def workers_for(n):
    """Split the CPUs between n trainings running at once."""
    return max(2, (os.cpu_count() or 8) // n)

def parallel(jobs, n):
    """jobs: {name: fn(log)}. Run at most n at once on the one GPU; the others finish even if one fails, and a re-run
    of the cell skips what is done."""
    os.makedirs('/content/logs', exist_ok=True)
    start = time.time()
    def work(name, fn):
        log, t = f'/content/logs/{name}.log', time.time()
        try:
            fn(log)
            print(f'Xong {name}: {(time.time() - t) / 60:.1f} phút', flush=True)
        except Exception:
            print(f'LỖI {name}, 30 dòng cuối của {log}:', flush=True)
            subprocess.run(['tail', '-n', '30', log])
            raise
    with ThreadPoolExecutor(n) as pool:
        futures = [pool.submit(work, k, v) for k, v in jobs.items()]
    errors = [f.exception() for f in futures if f.exception()]
    print(f'Tổng {(time.time() - start) / 60:.1f} phút cho {len(jobs)} job, tối đa {n} job cùng lúc')
    assert not errors, errors[0]

def decode_job(*args, **kw):
    return lambda log: evaluate_once(*args, log=log, **kw)

def eval_name(split, beams=None, bias=None, n=30, sources=None, lexicon=None):
    """Same output name as scripts/evaluate.py."""
    name = split + (f'_beam{beams}' if beams else '')
    if bias:
        name += f'_bias-{bias}' + (str(n) if bias != 'video' else '') + (f'-{os.path.basename(lexicon)[:-4]}' if lexicon else '')
    if sources:
        name += '_' + '-'.join(sources)
    return name

def metrics(exp, name):
    p = f'{PROJECT}/experiments/{exp}/eval/{name}/metrics.json'
    return json.load(open(p)) if os.path.exists(p) else None

def evaluate_once(exp, split, beams=None, bias=None, n=30, sources=None, lexicon=None, log=None):
    """Decode once; skip if the output already exists (TEST/HARD/probe are never re-run)."""
    name = eval_name(split, beams, bias, n, sources, lexicon)
    if metrics(exp, name):
        print('Đã có, bỏ qua:', exp, name); return
    run('scripts/evaluate.py', exp, '--split', split,
        *(['--allow-test'] if split in ('test', 'hard', 'probe') else []),
        *(['--num-beams', beams] if beams else []),
        *(['--bias', bias, '--distractors', n] if bias else []),
        *(['--lexicon', lexicon] if lexicon else []),
        *(['--sources', *sources] if sources else []), log=log)

def train_once(exp, log=None, workers=None):
    """Train unless experiments/<exp>/final exists; resume from checkpoints; drop checkpoints afterwards."""
    final = f'{PROJECT}/experiments/{exp}/final/adapter_model.safetensors'
    if os.path.exists(final):
        print('Đã train, bỏ qua:', exp); return
    run('scripts/train.py', exp, *(['--num-workers', workers] if workers else []), log=log)
    assert os.path.exists(final), f'không thấy {final}'
    shutil.rmtree(f'{PROJECT}/experiments/{exp}/checkpoints', ignore_errors=True)

def official(exp, log=None, workers=None):
    """Fixed-recipe model on the official split: DEV, TEST, HARD once each, greedy."""
    train_once(exp, log, workers)
    for split in ['dev', 'test', 'hard']:
        evaluate_once(exp, split, log=log)

def official_job(exp, n):
    return lambda log: official(exp, log, workers_for(n))

def probe_job(exp, n):
    def job(log):
        train_once(exp, log, workers_for(n))
        evaluate_once(exp, 'probe', log=log)
    return job
''')
md('''## A1. Nghiên cứu rò rỉ, PhoWhisper-small, 3 seed (~3 giờ)
Hai bộ Train có **cùng số clip** (dựng bởi `scripts/build_leak_data.py`); khác nhau đúng ở 722 clip: bản rò rỉ lấy các clip
đó từ chính những video có trong tập probe. Chấm trên cùng tập probe: `probe_seen` (video có clip bị rò rỉ) và
`probe_unseen` (đối chứng, không rò rỉ ở cả hai bản). Rò rỉ thật thì bản rò rỉ chỉ tốt hơn ở `probe_seen`.
''')
code('''# 3 job cùng lúc; xem tiến độ từng job ở /content/logs/
sh('python', '-c', "from huggingface_hub import snapshot_download as d; d('vinai/PhoWhisper-small')")
parallel({exp: probe_job(exp, 3) for exp in (f'L_small_{c}_s{s}' for s in [13, 14, 15] for c in ['clean', 'leak'])}, 3)
''')
md('''## B1. Ablation nguồn dữ liệu trên split chính thức + O2 (~4,5 giờ)
- D1: small + LoRA **chỉ ViMedCSS**, đúng thiết lập "PhoWhisper-Small + LoRA" của paper (TEST 27,13).
- D2: D1 + VietMed. D3: đủ 4 nguồn. Cùng công thức nên khác biệt chỉ do dữ liệu.
- O2: O3W bỏ oversampling (đo riêng tác dụng oversampling). O3W giải mã greedy thêm trên TEST/HARD để so cùng cách giải mã.
''')
code('''sh('python', '-c', "from huggingface_hub import snapshot_download as d; d('vinai/PhoWhisper-small'); d('openai/whisper-large-v3')")
O3W = 'O3W_whisper_large_v3_lora_official_termos'
parallel({exp: official_job(exp, 3) for exp in ['D1_small_lora_vimedcss_official', 'D2_small_lora_vimedcss_vietmed_official',
                                                'D3_small_lora_all4_official']}, 3)
official('O2_whisper_large_v3_lora_official')
parallel({f'O3W_{s}': decode_job(O3W, s) for s in ['test', 'hard']}, 2)  # O3W đã chốt beam 5; greedy chỉ để so ablation cùng cách giải mã
''')
md('''## C1. Huấn luyện có prompt thuật ngữ + giải mã có danh sách thuật ngữ (~5 giờ)
C1 = O3W + prompt thuật ngữ khi train (một biến). Sau đó giải mã ViMedCSS TEST và HARD của **cả C1 và O3W** với 4 kiểu danh sách:
- `oracle30`: thuật ngữ của clip + 30 thuật ngữ gây nhiễu (giao thức chuẩn của các bài contextual biasing);
- `distract30`: chỉ 30 thuật ngữ gây nhiễu (đo việc chèn nhầm thuật ngữ);
- `video`: mọi thuật ngữ của video đó (giống danh sách từ khóa của một bài giảng);
- `retrieve30-mesh2026`: **kết quả triển khai chính**: giải mã 2 lượt, chọn 30 mục gần nhất từ từ điển độc lập (từ vựng Train + MeSH 2026), không dùng nhãn của clip;
- `retrieve30`: giống trên nhưng từ điển chứa sẵn mọi thuật ngữ của tập chấm: chỉ là **mức trần**.

Thêm `oracle10` và `oracle50` trên HARD để đo tác động của độ dài danh sách.
So O3W (không train với prompt) với C1 cho biết việc train có prompt có cần thiết không.
Ô này chạy smoke test 2 bước trước khi train, để lỗi (nếu có) hiện ra ngay thay vì sau 2 giờ.
''')
code('''C1 = 'C1_whisper_large_v3_lora_official_termos_prompt'
if not os.path.exists(f'{PROJECT}/experiments/{C1}/final/adapter_model.safetensors'):
    run('scripts/train.py', C1, '--smoke')
official(C1)
O3W = 'O3W_whisper_large_v3_lora_official_termos'
parallel({f'O3W_{s}': decode_job(O3W, s) for s in ['hard', 'test']}, 2)  # lượt đầu của `retrieve`
jobs, src = {}, dict(sources=['ViMedCSS'])
for tag, exp in [('C1', C1), ('O3W', O3W)]:
    for split in ['hard', 'test']:
        for bias in ['oracle', 'distract', 'video', 'retrieve']:
            jobs[f'{tag}_{split}_{bias}'] = decode_job(exp, split, bias=bias, **src)
        jobs[f'{tag}_{split}_mesh'] = decode_job(exp, split, bias='retrieve', lexicon='data/lexicons/mesh2026.txt', **src)
    for n in [10, 50]:
        jobs[f'{tag}_hard_oracle{n}'] = decode_job(exp, 'hard', bias='oracle', n=n, **src)
parallel(jobs, 6)                                    # giải mã có danh sách chạy batch 1, chia GPU được nhiều nhất
''')
md('''## A2. Rò rỉ với mô hình lớn hơn (~6,5 giờ)
''')
code('''sh('python', '-c', "from huggingface_hub import snapshot_download as d; d('vinai/PhoWhisper-medium'); d('openai/whisper-large-v3')")
parallel({exp: probe_job(exp, 2) for exp in (f'L_{b}_{c}_s13' for b in ['medium', 'largev3'] for c in ['clean', 'leak'])}, 2)
''')
md('''## B2. Backbone, đối chứng oversampling, O3P (~7 giờ)
O2R lặp số clip bằng O3W nhưng chọn ngẫu nhiên: cùng số bước cập nhật, nên O2R → O3W đo riêng việc nhắm vào thuật ngữ hiếm.
O3P là ứng viên vượt PhoWhisper trên VIVOS (4,67). Cách giải mã (greedy hay beam 5) được chọn trên **DEV** và ghi lại
trước khi giải mã TEST.
''')
code('''sh('python', '-c', "from huggingface_hub import snapshot_download as d; d('openai/whisper-large-v3')")
parallel({exp: official_job(exp, 2) for exp in ['D4_whisper_large_v3_lora_vimedcss_official', 'O2R_whisper_large_v3_lora_official_randos']}, 2)
O3P = 'O3P_phowhisper_large_lora_official_termos'
train_once(O3P)
evaluate_once(O3P, 'dev')
evaluate_once(O3P, 'dev', beams=5)
choice_file = f'{PROJECT}/reports/paper/o3p_decode_choice.json'
if not os.path.exists(choice_file):
    g, b = metrics(O3P, 'dev'), metrics(O3P, 'dev_beam5')
    choice = dict(greedy_dev_wer=g['overall']['wer'], beam5_dev_wer=b['overall']['wer'],
                  num_beams=5 if b['overall']['wer'] < g['overall']['wer'] else None,
                  rule='lower official-DEV WER; written before any O3P TEST/HARD decode')
    assert metrics(O3P, 'test') is None and metrics(O3P, 'test_beam5') is None
    json.dump(choice, open(choice_file, 'w'), indent=1)
choice = json.load(open(choice_file))
print('O3P giải mã:', choice)
for split in ['hard', 'test']:
    evaluate_once(O3P, split, beams=choice['num_beams'])
''')
md('''## Xem nhanh kết quả
Bảng đầy đủ (khoảng tin cậy bootstrap theo video, CS-WER, recall theo độ hiếm) tính ở máy local bằng
`scripts/build_paper_tables.py` sau khi tải kết quả về.
''')
code('''import pandas as pd
rows = []
for exp in sorted(os.listdir(f'{PROJECT}/experiments')):
    for name in sorted(os.listdir(f'{PROJECT}/experiments/{exp}/eval')) if os.path.isdir(f'{PROJECT}/experiments/{exp}/eval') else []:
        m = metrics(exp, name)
        if m and (exp[:2] in ('L_', 'D1', 'D2', 'D3', 'D4', 'C1') or exp.startswith(('O2', 'O3'))):
            for src, v in m['by_source'].items():
                rows.append(dict(exp=exp, eval=name, source=src, wer=round(100 * v['wer'], 2),
                                 term_recall=None if v.get('cs_term_recall') is None else round(100 * v['cs_term_recall'], 1)))
pd.set_option('display.max_rows', 300)
display(pd.DataFrame(rows))
''')
md('''## Tải kết quả về máy
Gói `eval/`, `runs.jsonl`, `config.yaml` của mọi thí nghiệm paper và `reports/paper/` (không gồm `final/` và checkpoint;
adapter vẫn nằm trên Drive). Giải nén vào `ASR/` ở máy local rồi báo để tính bảng.
''')
code('''import zipfile
from google.colab import files
out = f'{PROJECT}/paper_results.zip'
keep = lambda e: e[:2] in ('L_', 'D1', 'D2', 'D3', 'D4', 'C1') or e.startswith(('O2', 'O3'))
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for exp in filter(keep, os.listdir(f'{PROJECT}/experiments')):
        base = f'{PROJECT}/experiments/{exp}'
        for root, dirs, fs in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ('final', 'checkpoints', 'smoke')]
            for f in fs:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), PROJECT))
    for root, _, fs in os.walk(f'{PROJECT}/reports/paper'):
        for f in fs:
            z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), PROJECT))
print(out, round(os.path.getsize(out) / 2**20, 1), 'MB')
files.download(out)
''')
for i, cell in enumerate(cells):
    cell['id'] = f'colab-paper-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'accelerator': 'GPU', 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1] / 'colab_paper.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
