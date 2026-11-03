"""Regenerate the UTF-8 A1 (leakage study, PhoWhisper-small) laptop notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 14 — A1: nghiên cứu rò rỉ video trên laptop (PhoWhisper-small, 3 seed)

Hai bộ Train cùng 29.913 clip, chỉ khác nguồn gốc 722 clip. Chấm trên cùng tập `probe`: `probe_seen` (video có clip trong Train rò rỉ)
và `probe_unseen` (đối chứng). Kế hoạch và tiêu chí đọc kết quả: `reports/paper/PAPER_PLAN.md`.

- **Cấu hình đã đo trên máy này:** micro-batch 4 × tích lũy 8 (batch hiệu dụng vẫn 32), 8,5 giây/bước, GPU tổng 3,35 GB, RAM 1,6 GB.
- **Thời gian ước tính:** 2.802 bước ≈ 6,6 giờ train, thêm eval và giải mã probe khoảng 1 giờ: **gần 8 giờ mỗi lần chạy, khoảng 16 giờ mỗi cặp, 47 giờ cho cả 6 lần**.
- **Không trộn máy:** cả 6 lần chạy A1 phải cùng trên laptop. Đã chạy ở đây thì không chạy ô A1 trên Colab nữa.

Trước khi train: tắt kernel các notebook khác, đóng Chrome, ChatGPT, Zalo, cắm sạc, tắt sleep. Mỗi lần chỉ một job GPU.
Chạy lại ô train sẽ tiếp tục từ checkpoint gần nhất (lưu 10 phút một lần) và bỏ qua phần đã xong. Giữ nguyên `RUN_ID` và `DUTY` khi chạy lại.
''')
code('''from pathlib import Path
import sys, json, ctypes
here = Path.cwd().resolve()
PROJECT = next((p for p in [here, *here.parents] if (p/'asr/config.py').is_file()), None)
if PROJECT is None:
    PROJECT = next((p/'ASR' for p in [here, *here.parents] if (p/'ASR/asr/config.py').is_file()), None)
assert PROJECT is not None, 'Mở notebook từ trong workspace dự án'
PYTHON = PROJECT.parent/'.venv-asr/Scripts/python.exe'
assert PYTHON.is_file(), 'Chạy ASR/local/setup.ps1 trước'
sys.path.insert(0, str(PROJECT/'local'))
from notebook_process import run_cli as _run_cli
def run_cli(arguments):
    return _run_cli(PROJECT, PYTHON, arguments)
RUNNER = PROJECT/'scripts/local_run.py'
RUNS = PROJECT/'local/runs'
RUN_ID, PROBE_ID, DUTY = 'local_a1', 'probe_a1', '1.0'
assert (PROJECT/'data/ViMedCSS_leak/probe.parquet').is_file(), 'Thiếu data/ViMedCSS_leak: chạy scripts/build_leak_data.py'

class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

def commit_free_gb():
    s = _MemStatus(); s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPageFile / 1e9

def gpu_job_running():
    """PID của job GPU local đang chạy (giữ local/gpu.lock), hoặc None."""
    import psutil
    lock = PROJECT/'local/gpu.lock'
    if not lock.exists():
        return None
    pid = int(lock.read_text())
    return pid if psutil.pid_exists(pid) else None

def adapter(exp):
    return RUNS/exp/RUN_ID/'final'

def probe_done(exp):
    return (RUNS/exp/PROBE_ID/'eval/probe/metrics.json').is_file()

def run_seeds(seeds):
    """Train rồi giải mã probe một lần cho từng thí nghiệm; phần đã xong được bỏ qua."""
    for seed in seeds:
        for cond in ['clean', 'leak']:
            exp = f'L_small_{cond}_s{seed}'
            if probe_done(exp):
                print('Đã xong, bỏ qua:', exp, flush=True)
                continue
            pid = gpu_job_running()
            if pid:
                print(f'Đang có job GPU khác (PID {pid}); dừng lại, không chạy thêm.')
                return
            if commit_free_gb() < 4:
                print(f'Bộ nhớ commit chỉ còn {commit_free_gb():.1f} GB; tắt bớt ứng dụng rồi chạy lại ô này.')
                return
            if not (adapter(exp)/'adapter_model.safetensors').is_file():
                run_cli([RUNNER, 'train', exp, '--run-id', RUN_ID, '--duty-cycle', DUTY])
            run_cli([RUNNER, 'eval', exp, '--run-id', PROBE_ID, '--split', 'probe', '--allow-test',
                     '--checkpoint', adapter(exp), '--duty-cycle', DUTY])
    print('Xong các seed', seeds)

print('Job GPU đang chạy:', gpu_job_running() or 'không')
free = commit_free_gb()
print(f'Bộ nhớ commit trống: {free:.1f} GB', '' if free >= 4 else '-> THẤP: đóng bớt ứng dụng rồi chạy lại ô này')
for s in [13, 14, 15]:
    for c in ['clean', 'leak']:
        print(f'L_small_{c}_s{s}:', 'xong' if probe_done(f'L_small_{c}_s{s}') else 'chưa xong')
''')
code("run_cli([RUNNER, 'check', 'L_small_leak_s13', '--run-id', RUN_ID])\n")
md('''## 1. Seed 13 (cặp sạch và rò rỉ, khoảng 16 giờ)

Đây là ô cần chạy đầu tiên. Xong thì chạy ô **Xem kết quả** rồi gửi bảng đó.
''')
code("run_seeds([13])\n")
md('''## Theo dõi (chỉ đọc file, chạy được khi đang train ở ô khác)
''')
code('''def latest_state(run):
    ckpts = sorted((run/'checkpoints').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    return json.loads((ckpts[-1]/'trainer_state.json').read_text(encoding='utf-8')) if ckpts else None

for s in [13, 14, 15]:
    for c in ['clean', 'leak']:
        exp = f'L_small_{c}_s{s}'
        st = latest_state(RUNS/exp/RUN_ID)
        if st:
            print(exp, f"bước {st['global_step']}/{st['max_steps']}",
                  {e['step']: round(e['eval_wer'], 4) for e in st['log_history'] if 'eval_wer' in e})
''')
md('''## Xem kết quả

Chỉ dùng CPU. `did` là DiD, `did_lo` và `did_hi` là khoảng tin cậy 95% theo video. Nếu `did` của seed 13 không dương, được dừng ở đây và
báo là kết quả thử một seed (`PAPER_PLAN.md`). Nếu dương, chạy tiếp ô dưới: kết luận cần đủ 3 seed.
''')
code("run_cli([PROJECT/'scripts/build_paper_tables.py', '--leak-only'])\n")
md('''## 2. Seed 14 và 15 (khoảng 31 giờ)
''')
code("run_seeds([14, 15])\n")
for i, cell in enumerate(cells):
    cell['id'] = f'a1-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/14_A1_LOCAL.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
