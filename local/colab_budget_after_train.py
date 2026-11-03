# Ô dán vào colab_budget.ipynb SAU KHI ô train in "saved best model" (thay các ô 2–5 gốc, vốn cần ~4,5 giờ GPU).
# Chọn greedy/beam 5 trên DEV con cố định (300 clip/nguồn), chốt, rồi TEST/HARD hệ cuối + HARD zero-shot B3/B2.
!pkill -f "[s]cripts/evaluate.py"   # tắt tiến trình giải mã còn chạy ngầm sau khi bấm dừng
import time, hashlib, datetime, yaml, pandas as pd
EXP = 'O3W_whisper_large_v3_lora_official_termos'
T0 = time.time()
def step(msg): print(f'\n[{(time.time() - T0) / 60:.0f} phút] {msg}', flush=True)

# 1) DEV con: greedy vs beam 5 (chỉ dùng DEV để chọn)
rows = []
for beams in [None, 5]:
    name = 'dev' + (f'_beam{beams}' if beams else '') + '_sub300'
    if not dev_metrics(EXP, name):
        step(f'DEV con, beam {beams or 1}')
        run('scripts/evaluate.py', EXP, '--split', 'dev', '--max-per-source', 300, *(['--num-beams', beams] if beams else []))
    m = dev_metrics(EXP, name)
    rows.append(dict(exp=EXP, decode=name, num_beams=beams or 1, wer=100*m['overall']['wer'],
                     vimedcss=100*m['by_source']['ViMedCSS']['wer'], term_recall=100*m['overall']['cs_term_recall'],
                     truncated=m['overall']['n_truncated']))
table = pd.DataFrame(rows).sort_values('wer')
display(table.round(2))
FINAL = table.iloc[0].to_dict()

# 2) Chốt (freeze) TRƯỚC khi giải mã TEST
out = f'{PROJECT}/reports/final_freeze_official'
if not os.path.exists(f'{out}/freeze.json'):
    assert not any(os.path.exists(f'{PROJECT}/experiments/{EXP}/eval/{s}') for s in ('test', 'test_beam5', 'hard', 'hard_beam5')), 'đã có TEST'
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    record = dict(frozen_at=datetime.datetime.now().isoformat(timespec='seconds'), final_system=EXP,
                  num_beams=int(FINAL['num_beams']),
                  config=yaml.safe_load(open(f'{PROJECT}/experiments/{EXP}/config.yaml')),
                  adapter_sha256=sha(f'{PROJECT}/experiments/{EXP}/final/adapter_model.safetensors'),
                  selection='lowest WER on a fixed official-DEV subset (300 clips/source, config seed) of O3W greedy vs beam 5; TEST not decoded yet',
                  dev_table=table.round(4).to_dict('records'))
    os.makedirs(out, exist_ok=True)
    json.dump(record, open(f'{out}/freeze.json', 'w'), ensure_ascii=False, indent=1)
freeze = json.load(open(f'{out}/freeze.json'))
print('Đã chốt:', freeze['final_system'], 'beam', freeze['num_beams'])

# 3) TEST + HARD hệ cuối (mỗi thứ một lần), rồi HARD zero-shot để so cách chấm với paper
beams = 5 if freeze['num_beams'] == 5 else None
for split in ['hard', 'test']:
    step(f'{split} — hệ cuối')
    evaluate_once(freeze['final_system'], split, beams)
for exp in ['B3_whisper_large_v3_zeroshot', 'B2_phowhisper_large_zeroshot']:
    step(f'hard — {exp}')
    evaluate_once(exp, 'hard')
step('XONG')

# 4) Bảng kết quả (WER %) cạnh số paper
sfx = '_beam5' if beams else ''
t, h = dev_metrics(freeze['final_system'], 'test' + sfx), dev_metrics(freeze['final_system'], 'hard' + sfx)
print('HỆ CUỐI:', {'ViMedCSS TEST': round(100*t['by_source']['ViMedCSS']['wer'], 2), 'ViMedCSS HARD': round(100*h['overall']['wer'], 2),
                   'VIVOS test': round(100*t['by_source']['VIVOS']['wer'], 2), 'VietMed test': round(100*t['by_source']['VietMed']['wer'], 2),
                   'FLEURS test': round(100*t['by_source']['FLEURS']['wer'], 2),
                   'term recall TEST': round(100*t['by_source']['ViMedCSS']['cs_term_recall'], 2)})
for exp in ['B3_whisper_large_v3_zeroshot', 'B2_phowhisper_large_zeroshot']:
    print(exp, 'HARD WER:', round(100*dev_metrics(exp, 'hard')['overall']['wer'], 2))
print('Paper: ViMedCSS TEST best 23.67 (AG), HARD best 33.73; zero-shot HARD large-v3 39.35, PhoWhisper-large 37.37; '
      'VIVOS PhoWhisper-large 4.67; VietMed XLSR-53-Viet 29.6')
