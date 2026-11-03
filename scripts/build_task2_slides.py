"""Build the Task 2 slide deck (python-pptx) from the report tables/figures. Run after build_task2_report.py.

Run from ASR/:  ..\.venv-asr\Scripts\python.exe scripts/build_task2_slides.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/task2_20261030"
FIG, TAB = OUT / "figures", OUT / "tables"

DARK, TEAL, MINT = RGBColor(0x0B, 0x4F, 0x5C), RGBColor(0x1C, 0x7C, 0x8C), RGBColor(0xE8, 0xF3, 0xF4)
CORAL, TEXT, MUTED, WHITE = RGBColor(0xD9, 0x4F, 0x2B), RGBColor(0x1F, 0x2A, 0x30), RGBColor(0x5B, 0x6B, 0x73), RGBColor(255, 255, 255)
GREEN, GREY = RGBColor(0x2E, 0x8B, 0x57), RGBColor(0xF4, 0xF6, 0xF7)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5
state = {"n": 0}


def fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None):
    """paras: list of dict(text|runs, size, bold, color, bullet, align, font, after)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = p.get("runs") or [dict(text=p["text"], bold=p.get("bold", False), color=p.get("color", TEXT))]
        for r in runs:
            run = para.add_run()
            run.text = r["text"]
            run.font.size = Pt(p.get("size", 16))
            run.font.bold = r.get("bold", p.get("bold", False))
            run.font.italic = r.get("italic", False)
            run.font.name = p.get("font", BODY)
            run.font.color.rgb = r.get("color", p.get("color", TEXT))
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        para.space_after = Pt(p.get("after", 6))
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.28))))
            pPr.set("indent", str(-int(Inches(0.28))))
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "•")
    return tb


def new_slide(title, dark=False, section=None):
    s = prs.slides.add_slide(BLANK)
    state["n"] += 1
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = DARK if dark else WHITE
    if title:
        textbox(s, 0.6, 0.35, 12.1, 0.95, [dict(text=title, size=30, bold=True, color=WHITE if dark else DARK, font=HEAD)],
                anchor=MSO_ANCHOR.MIDDLE, name="Title")
    if section:
        textbox(s, 0.6, 0.12, 8, 0.3, [dict(text=section.upper(), size=11, bold=True, color=TEAL if not dark else MINT)], name="Section")
    textbox(s, 11.9, 7.0, 0.9, 0.3, [dict(text=str(state["n"]), size=11, color=MUTED, align=PP_ALIGN.RIGHT)], name="SlideNo")
    return s


def card(slide, x, y, w, h, color=MINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    r = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    fill(r, color)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        r.adjustments[0] = 0.08
    return r


def stat(slide, x, y, w, h, big, label, color=TEAL, bg=MINT, big_size=34):
    card(slide, x, y, w, h, bg)
    textbox(slide, x + 0.1, y + 0.08, w - 0.2, h * 0.55, [dict(text=big, size=big_size, bold=True, color=color, font=HEAD, align=PP_ALIGN.CENTER)],
            anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, x + 0.1, y + h * 0.58, w - 0.2, h * 0.4, [dict(text=label, size=13, color=TEXT, align=PP_ALIGN.CENTER)],
            anchor=MSO_ANCHOR.TOP)


def image(slide, name, x, y, w, h):
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - dw) / 2), Inches(y + (h - dh) / 2), Inches(dw), Inches(dh))
    pic.name = name
    return pic


def table(slide, x, y, w, rows, widths, size=13, header_size=None, row_h=0.4, bold_first_col=False, highlight=None, align_right=()):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = shape.table
    tot = sum(widths)
    for i, wd in enumerate(widths):
        tbl.columns[i].width = Inches(w * wd / tot)
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    for r in range(nr):
        tbl.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = DARK
            elif highlight and r in highlight:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xEC, 0xE5)
            else:
                cell.fill.fore_color.rgb = GREY if r % 2 == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(rows[r][c])
            run.font.size = Pt(header_size or size) if r == 0 else Pt(size)
            run.font.name = BODY
            run.font.bold = r == 0 or (bold_first_col and c == 0) or bool(highlight and r in highlight)
            run.font.color.rgb = WHITE if r == 0 else TEXT
            p.alignment = PP_ALIGN.RIGHT if (c in align_right and r > 0) else PP_ALIGN.LEFT
    return shape


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def fmt(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


# ------------------------------------------------------------------ data
dev3 = pd.read_csv(TAB / "ablation_dev_v3.csv").set_index("run")
devo = pd.read_csv(TAB / "ablation_dev_official.csv").set_index("run")
lat = pd.read_csv(TAB / "latency_size.csv").set_index("system")
imb = json.loads((TAB / "imbalance.json").read_text(encoding="utf-8"))
fail = json.loads((TAB / "failure_summary.json").read_text(encoding="utf-8"))
terms = json.loads((TAB / "term_summary.json").read_text(encoding="utf-8"))

# ================================================================== 1 title
s = new_slide(None, dark=True)
textbox(s, 0.9, 1.5, 11.5, 2.2, [
    dict(text="Nhận dạng giọng nói tiếng Việt y khoa có chèn từ tiếng Anh", size=40, bold=True, color=WHITE, font=HEAD, after=10),
    dict(text="PhoWhisper / Whisper-large-v3 + LoRA trên ViMedCSS, VietMed, VIVOS và FLEURS", size=22, color=MINT)], name="Title")
textbox(s, 0.9, 4.2, 11.5, 1.0, [
    dict(text="Báo cáo tiến độ Task 2: xử lý dữ liệu, thực nghiệm Baseline và mô hình đề xuất", size=20, bold=True, color=WHITE)])
textbox(s, 0.9, 5.5, 11.5, 1.2, [
    dict(text="Mã nhóm: ........   Sinh viên: ........................   GVHD: ........................", size=16, color=MINT, after=4),
    dict(text="Môn học Deep Learning, FPT, học kỳ Fall 2026", size=14, color=MINT)])
notes(s, "Điền mã nhóm, tên sinh viên và giảng viên hướng dẫn vào dòng thông tin trước khi trình bày.")

# ================================================================== 2 agenda
s = new_slide("Nội dung báo cáo")
items = [("I", "Xử lý và phân tích dữ liệu", "4 corpus, làm sạch, mất cân bằng, chống rò rỉ video"),
         ("II", "Thực nghiệm 2 mô hình Baseline", "PhoWhisper-small và Whisper-small zero-shot, Paper vs Thực nghiệm"),
         ("III", "Phát triển mô hình đề xuất V1 → V5B và O3W", "LoRA, SpecAugment, beam search, backbone lớn, oversampling thuật ngữ; so trực tiếp với paper"),
         ("IV", "Phân tích lỗi và tính khả thi", "Failure cases, độ chính xác, tốc độ, kích thước")]
for i, (num, head, sub) in enumerate(items):
    y = 1.65 + i * 1.3
    card(s, 0.9, y, 11.5, 1.1, MINT)
    c = card(s, 1.1, y + 0.2, 0.7, 0.7, TEAL, MSO_SHAPE.OVAL)
    textbox(s, 1.1, y + 0.2, 0.7, 0.7, [dict(text=num, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER, font=HEAD)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 2.1, y + 0.12, 10, 0.5, [dict(text=head, size=22, bold=True, color=DARK, font=HEAD)])
    textbox(s, 2.1, y + 0.58, 10, 0.4, [dict(text=sub, size=15, color=MUTED)])

# ================================================================== 3 sources
s = new_slide("Thu thập và tổng quan dữ liệu gốc", section="Chương I · Dữ liệu")
rows = [["Nguồn", "Loại", "Clip train / dev / test", "Giờ", "Dung lượng", "Đặc điểm"],
        ["VIVOS", "Online", "11.476 / 184 / 760", "15,7", "915 MB", "Đọc văn bản, nhiều người nói"],
        ["FLEURS vi_vn", "Online", "2.994 / 361 / 857", "13,3", "627 MB", "Câu đọc, miền tổng hợp"],
        ["VietMed", "Online", "2.773 / 2.912 / 3.437", "15,8", "699 MB", "Thoại y khoa thực tế"],
        ["ViMedCSS", "Online", "11.832 / 1.714 / 1.614 + hard 658", "30,8", "1,9 GB", "Y khoa Việt–Anh, có nhãn thuật ngữ"]]
table(s, 0.6, 1.55, 12.1, rows, [1.6, 0.9, 3.2, 0.7, 1.2, 3.4], size=14, row_h=0.55, bold_first_col=True)
card(s, 0.6, 4.55, 5.9, 2.2, MINT)
textbox(s, 0.8, 4.65, 5.5, 2.0, [
    dict(text="Định dạng thống nhất", size=18, bold=True, color=DARK, font=HEAD),
    dict(text="Audio FLAC 16 kHz mono, gói trong parquet", size=15, bullet=True),
    dict(text="Trường: id, source, split, group_id, speaker_id, text, cs_terms, duration, audio", size=15, bullet=True),
    dict(text="Kèm sha256 từng split (registry.json)", size=15, bullet=True)])
card(s, 6.8, 4.55, 5.9, 2.2, MINT)
textbox(s, 7.0, 4.65, 5.5, 2.0, [
    dict(text="Ví dụ mẫu ViMedCSS (có cs_terms)", size=18, bold=True, color=DARK, font=HEAD),
    dict(text="“…lấy một lượng vaseline chúng ta bôi lên môi…”", size=15, bullet=True),
    dict(text="Thuật ngữ: vaseline, virus, glucose, hormone…", size=15, bullet=True),
    dict(text="Toàn bộ 4 nguồn là dữ liệu công khai; không có dữ liệu Offline", size=15, bullet=True)])
notes(s, "Nguồn VIVOS/FLEURS/VietMed/ViMedCSS. Không thu thập Offline. Số liệu từ data/registry.json.")

# ================================================================== 4 cleaning
s = new_slide("Làm sạch dữ liệu và gán nhãn", section="Chương I · Dữ liệu")
steps = [("Chuẩn hóa audio", "16 kHz, mono, FLAC; bỏ clip ngoài (0, 30 s]"),
         ("Chuẩn hóa văn bản", "NFC, chữ thường, bỏ dấu câu; cùng luật cho nhãn và chấm điểm"),
         ("Loại trùng", "4 clip trùng nội dung"),
         ("Loại nhiễu", "64 clip im lặng (đỉnh biên độ < 1e-3)"),
         ("Loại nhãn lệch", "14 clip nhãn > 16 token/giây khỏi Train"),
         ("Chống rò rỉ", "Chia lại ViMedCSS theo video, 0 video trùng")]
w, gap = 1.88, 0.16
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * (w + gap)
    card(s, x, 1.7, w, 2.9, MINT)
    c = card(s, x + w / 2 - 0.3, 1.85, 0.6, 0.6, TEAL, MSO_SHAPE.OVAL)
    textbox(s, x + w / 2 - 0.3, 1.85, 0.6, 0.6, [dict(text=str(i + 1), size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, x + 0.08, 2.55, w - 0.16, 0.7, [dict(text=h, size=16, bold=True, color=DARK, align=PP_ALIGN.CENTER, font=HEAD)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, x + 0.08, 3.25, w - 0.16, 1.3, [dict(text=d, size=13, color=TEXT, align=PP_ALIGN.CENTER)])
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + w - 0.02), Inches(3.0), Inches(gap + 0.04), Inches(0.3))
        fill(a, CORAL)
card(s, 0.6, 4.95, 12.1, 1.85, GREY)
textbox(s, 0.8, 5.05, 11.7, 1.7, [
    dict(text="Gán nhãn", size=18, bold=True, color=DARK, font=HEAD),
    dict(text="Không gán nhãn thủ công: transcript và nhãn thuật ngữ lấy từ chính các corpus công khai. Vì vậy không có chỉ số Kappa.", size=15, bullet=True),
    dict(text="Hệ quả: chất lượng nhãn không được đo; có dấu hiệu nhãn lệch audio ở VietMed và ViMedCSS (xem phần phân tích lỗi).", size=15, bullet=True),
    dict(text="Giá trị thiếu: speaker_id rỗng ở FLEURS và ViMedCSS, chỉ dùng group_id để chống rò rỉ.", size=15, bullet=True)])

# ================================================================== 5 stats / imbalance
s = new_slide("Thống kê mô tả và mất cân bằng dữ liệu", section="Chương I · Dữ liệu")
rows = [["Split", "Clip", "% clip", "% giờ", "% từ"],
        ["Train", "29.075", "71,1%", "69,3%", "68,7%"],
        ["Dev", "5.171", "12,6%", "13,3%", "14,2%"],
        ["Test", "6.668", "16,3%", "17,5%", "17,0%"]]
table(s, 0.6, 1.5, 5.6, rows, [1.2, 1.2, 1, 1, 1], size=14, row_h=0.45, bold_first_col=True, align_right=(1, 2, 3, 4))
textbox(s, 0.6, 3.4, 5.6, 0.4, [dict(text="Mất cân bằng đo trên 3 trục", size=18, bold=True, color=DARK, font=HEAD)])
stat(s, 0.6, 3.85, 1.8, 1.85, "55,6%", "số từ DEV là VietMed (Train chỉ 10,6%)", color=CORAL, big_size=26)
stat(s, 2.5, 3.85, 1.8, 1.85, "22,5%", "lượt thuật ngữ thuộc top-10 (831 thuật ngữ)", color=CORAL, big_size=26)
stat(s, 4.4, 3.85, 1.8, 1.85, "92,7%", "clip DEV ViMedCSS trùng video với Train", color=CORAL, big_size=26)
textbox(s, 0.6, 5.85, 5.6, 1.1, [
    dict(text="173 thuật ngữ chỉ xuất hiện 1 lần; 390 thuật ngữ dưới 5 lần (47%).", size=14, bullet=True),
    dict(text="Thuật ngữ chưa từng thấy: recall chỉ 20,8% so với 75,8% đã thấy.", size=14, bullet=True)])
image(s, "fig1_source_composition.png", 6.4, 1.45, 6.4, 2.6)
image(s, "fig3_term_long_tail.png", 6.4, 4.15, 6.4, 2.75)

# ================================================================== 6 imbalance handling
s = new_slide("Xử lý mất cân bằng và so sánh trước / sau", section="Chương I · Dữ liệu")
image(s, "fig5_leakage_before_after.png", 0.5, 1.45, 7.4, 2.8)
rows = [["Vấn đề", "Giải pháp", "Trước → Sau"],
        ["Rò rỉ video Train → Dev", "Chia lại theo video 80/10/10", "92,7% → 0%"],
        ["Clip trùng, im lặng, nhãn lệch", "Loại trước khi chia", "−4, −64, −14 clip"],
        ["Quá khớp, thuật ngữ hiếm", "Augmentation SpecAugment", "WER 18,26 → 17,96"],
        ["Miền VietMed áp đảo DEV", "Báo cáo WER theo nguồn", "by_source + recall"]]
table(s, 0.6, 4.45, 7.3, rows, [2.6, 2.6, 2.1], size=13, row_h=0.46)
card(s, 8.2, 1.5, 4.5, 3.55, MINT)
textbox(s, 8.4, 1.6, 4.1, 3.4, [
    dict(text="Điều chưa làm, nói thẳng", size=18, bold=True, color=CORAL, font=HEAD),
    dict(text="Chưa dùng class weighting hay focal loss.", size=15, bullet=True),
    dict(text="Oversampling thuật ngữ hiếm chỉ mới áp dụng ở O3W (407 clip lặp 3 lần), chưa có ablation riêng.", size=15, bullet=True),
    dict(text="Train vẫn trộn 4 nguồn theo tỷ lệ tự nhiên.", size=15, bullet=True),
    dict(text="Split Hard sau khi chia lại chỉ còn 71 clip (trước đó 658), yếu về thống kê.", size=15, bullet=True),
    dict(text="Đây là hướng cải tiến tiếp theo.", size=15, bullet=True, bold=True)])
stat(s, 8.2, 5.25, 4.5, 1.6, "92,7% → 0%", "clip DEV có video trùng với Train, trước và sau khi chia lại", color=GREEN, big_size=32)

# ================================================================== 7 environment
s = new_slide("Cấu hình môi trường thực nghiệm", section="Chương II · Baseline")
rows = [["Hạng mục", "Giá trị"],
        ["Phần cứng", "Laptop, GPU RTX 4050 6 GB (ngân sách 4,5 GB VRAM)"],
        ["Framework", "PyTorch 2.6 (CUDA 12.4), transformers 4.46, peft 0.13"],
        ["Seed", "13 cho mô hình, thứ tự dữ liệu và chia split"],
        ["Fine-tune", "LoRA r=32, α=64, dropout 0,05 trên q, k, v, out_proj, fc1, fc2"],
        ["Batch hiệu dụng", "32 (V5: micro-batch 2 × tích lũy 16), fp16, gradient checkpointing"],
        ["Giải mã", "greedy hoặc beam 5; guard chống lặp (token ≤ 12/giây + 20)"],
        ["Chuẩn hóa chấm điểm", "NFC, chữ thường, bỏ dấu câu; WER, CER, recall thuật ngữ"],
        ["O3W (Colab)", "A100 80 GB, bf16, batch 32, khoảng 2,1 giờ train"]]
table(s, 0.6, 1.55, 7.6, rows, [1.9, 5.7], size=14, row_h=0.55, bold_first_col=True)
rows = [["Phiên bản", "lr", "Epoch"],
        ["V1", "1e-3", "3"], ["V2", "1e-4", "2"], ["V3, V4", "3e-4", "3"], ["V5", "2e-4", "2"], ["O3W", "1e-4", "2"]]
table(s, 8.5, 1.55, 4.2, rows, [1.6, 1.2, 1.0], size=14, row_h=0.45, bold_first_col=True)
card(s, 8.5, 4.4, 4.2, 2.35, MINT)
textbox(s, 8.65, 4.5, 3.9, 2.2, [
    dict(text="Lưu ý", size=16, bold=True, color=DARK, font=HEAD),
    dict(text="Hai baseline là zero-shot nên không có lr, optimizer, epoch.", size=14, bullet=True),
    dict(text="Các biến khác giữa phiên bản được đổi có chủ đích và ghi ở bảng ablation.", size=14, bullet=True)])

# ================================================================== 8 baselines
s = new_slide("Hai Baseline: Paper so với Thực nghiệm của nhóm", section="Chương II · Baseline")
rows = [["Mô hình", "Nguồn", "Dataset", "WER %"],
        ["Baseline 1 · PhoWhisper-small", "Paper", "ViMedCSS TEST chính thức", "36,31"],
        ["", "Nhóm", "ViMedCSS TEST chia theo video", "30,02"],
        ["", "Paper", "VIVOS test", "6,33"],
        ["", "Nhóm", "VIVOS test", "6,28"],
        ["", "Paper", "VietMed-Test (wav2vec2 tốt nhất)", "29,6"],
        ["", "Nhóm", "VietMed TEST", "25,20"],
        ["Baseline 2 · Whisper-small", "Paper", "ViMedCSS TEST chính thức", "50,03"],
        ["", "Nhóm", "ViMedCSS DEV chính thức (không guard / guard)", "46,45 / 40,54"],
        ["", "Paper", "FLEURS vi (chưa tra được số)", "—"],
        ["", "Nhóm", "FLEURS DEV (không guard / guard)", "25,61 / 25,26"]]
table(s, 0.6, 1.5, 12.1, rows, [3.4, 1.2, 4.7, 1.6], size=14, row_h=0.44, align_right=(3,), highlight=(2, 4, 6, 8, 10))
textbox(s, 0.6, 6.45, 12.1, 0.8, [dict(
    text="Hàng tô màu là kết quả của nhóm. Số paper đọc lại từ arXiv ngày 30/10/2026: ViMedCSS 2602.12911, PhoWhisper 2406.02555, VietMed 2404.05659 (tác giả ghi số chỉ để tham khảo). B1 chưa giải mã TEST nên so với paper bằng DEV, chỉ mang tính chỉ báo.",
    size=12, color=MUTED)])

# ================================================================== 9 gap analysis
s = new_slide("Gap Analysis: vì sao kết quả lệch với công bố", section="Chương II · Baseline")
stat(s, 0.6, 1.5, 3.9, 1.5, "0,05 điểm", "VIVOS: 6,28% so với 6,33% của paper, pipeline đánh giá hợp lý", color=GREEN, big_size=28)
stat(s, 4.7, 1.5, 3.9, 1.5, "−6,3 điểm", "ViMedCSS zero-shot: 30,02% so với 36,31%", color=TEAL, big_size=28)
stat(s, 8.8, 1.5, 3.9, 1.5, "62,6%", "recall thuật ngữ V4B, thấp hơn mức CS-WER 19,5% của paper", color=CORAL, big_size=28)
causes = [("Dataset và split", "Nhóm chia lại ViMedCSS theo video và train thêm VIVOS, FLEURS, VietMed; paper chỉ dùng ViMedCSS. Split chính thức rò rỉ video nên số cũ lạc quan."),
          ("Tiền xử lý", "Dấu câu, viết hoa, nhãn không nhất quán (Train có cả insulin và inulin, vaccin và vắcxin) ảnh hưởng trực tiếp WER; paper không công bố đủ chi tiết."),
          ("Metric", "Paper dùng CS-WER trong đoạn code-switch; nhóm dùng WER toàn câu và recall thuật ngữ khớp chuỗi con, không tương đương."),
          ("Siêu tham số", "Ít ảnh hưởng ở baseline vì zero-shot; xuất hiện từ V1 (lr 1e-3 phân kỳ).")]
for i, (h, d) in enumerate(causes):
    x = 0.6 + (i % 2) * 6.1
    y = 3.35 + (i // 2) * 1.8
    card(s, x, y, 5.95, 1.65, MINT)
    textbox(s, x + 0.15, y + 0.08, 5.65, 1.5, [dict(text=h, size=17, bold=True, color=DARK, font=HEAD, after=3), dict(text=d, size=14)])
textbox(s, 0.6, 6.95, 11, 0.4, [dict(text="Kết luận: các bản V không so trực tiếp được với paper; vì vậy nhóm train thêm O3W trên đúng split chính thức (Chương III).", size=13, bold=True, color=CORAL)])

# ================================================================== 9b new papers
s = new_slide("So sánh với các paper mới (2025 – 2026)", section="Chương II · Baseline")
rows = [["Paper", "Kết quả công bố", "Kết quả của nhóm", "Mức so sánh"],
        ["ViMedCSS (02/2026)", "TEST WER: small zero-shot 36,31; LoRA 27,13; AG 23,67 (CS-WER 19,50). HARD: AG 33,73", "O3W, TEST/HARD chính thức: 11,40 / 16,42. (V5B, TEST chia theo video: 14,78)", "Cùng tập, khác cách chấm; nhóm train thêm 3 nguồn"],
        ["PhoASR, EACL Findings (03/2026)", "VietMed-L N-WER: PhoWhisper-small 7,90; PhoASR-469h 9,51", "VietMed TEST: B0V 25,20; V5B 19,39; O3W 20,43", "Không so được: test đã tinh chỉnh"],
        ["TSPC (09/2025, bản 03/2026)", "Vi–En đời thường: PhoWhisper-base 27,90; TSPC 19,06", "Không có số cùng tập", "Không so được (không phải y khoa)"],
        ["VietMed bản 4 (10/2026)", "Test WER: XLSR-53 51,8; XLSR-53-Viet 29,6", "VietMed TEST: V5B 19,39; O3W 20,43 (B2 zero-shot 24,04)", "Cùng tập test, khác kiến trúc"],
        ["PhoWhisper (2024)", "VIVOS: small 6,33; medium 4,97; large 4,67", "VIVOS test: chạy lại small 6,28, large 4,61; V5B 4,86; O3W 7,03", "Trực tiếp: tái lập trùng paper; V5B vượt medium, chưa vượt large"]]
table(s, 0.6, 1.45, 12.1, rows, [2.3, 3.9, 3.4, 2.5], size=13, row_h=0.72, bold_first_col=True)
card(s, 0.6, 5.95, 12.1, 1.0, MINT)
textbox(s, 0.8, 6.0, 11.7, 0.9, [
    dict(text="WER thấp hơn số công bố ở 2 / 3 paper so được: ViMedCSS (TEST và HARD) và VietMed. Chưa đạt PhoWhisper-large trên VIVOS (4,67; V5B 4,86, O3W 7,03).", size=14, bullet=True, after=2),
    dict(text="Đây là so sánh số, chưa chứng minh phương pháp tốt hơn: khác dữ liệu train, backbone và cách chấm.", size=14, bullet=True, after=0)])
notes(s, "Mọi số paper được đọc lại từ arXiv ngày 30/10/2026; xem NUMBER_AUDIT.md.")

# ================================================================== 9c same-metric table
s = new_slide("Các bản V so với paper, cùng bộ metric", section="Chương II · Baseline")
so = pd.read_csv(TAB / "same_metric_ours.csv")
sp_ = pd.read_csv(TAB / "same_metric_paper.csv")
def pr(name):
    r = sp_[(sp_.split == "TEST") & (sp_.system == name)].iloc[0]
    z = sp_[(sp_.split == "TEST") & (sp_.system == "PhoWhisper-Small zero-shot")].iloc[0]
    return r, 100 * (1 - r.cs_wer / z.cs_wer)
def ou(split, name):
    r = so[(so.split == split) & (so.system == name)].iloc[0]
    z = so[(so.split == split) & (so.system == "B0V")].iloc[0]
    return r, 100 * (1 - r.cs_wer / z.cs_wer)
rows = [["Hệ thống (ViMedCSS)", "WER %", "CER %", "Giảm CS-WER so với zero-shot"]]
for label, name in [("Paper · small zero-shot", "PhoWhisper-Small zero-shot"), ("Paper · small + LoRA", "PhoWhisper-Small + LoRA"),
                    ("Paper · small + Attention Guide (tốt nhất)", "PhoWhisper-Small + AG (best)")]:
    r, red = pr(name)
    rows.append([label, fmt(r.wer), fmt(r.cer), "—" if "zero" in name else f"{fmt(red, 1)}%"])
for label, sp, name in [("Nhóm · V4B small + beam, TEST chia theo video", "TEST", "V4B"), ("Nhóm · V5 medium, TEST chia theo video", "TEST", "V5"),
                        ("Nhóm · V5B medium + beam, TEST chia theo video", "TEST", "V5B")]:
    r, red = ou(sp, name)
    rows.append([label, fmt(r.wer), fmt(r.cer), f"{fmt(red, 1)}%"])
table(s, 0.6, 1.5, 12.1, rows, [5.2, 1.4, 1.4, 3.6], size=14, row_h=0.5, bold_first_col=False, align_right=(1, 2, 3), highlight=(4, 5, 6))
card(s, 0.6, 5.35, 12.1, 1.5, MINT)
textbox(s, 0.8, 5.4, 11.7, 1.4, [
    dict(text="WER và CER: số của V5B thấp hơn số công bố (14,78% / 10,90% so với 23,67% / 14,73%), một phần nhờ train thêm VIVOS, FLEURS, VietMed.", size=14, bullet=True, after=3),
    dict(text="Thuật ngữ y khoa: V5B giảm CS-WER 56,6%; paper: LoRA 51,6%, Attention Guide 68,8%, theo luật tính khác.", size=14, bullet=True, after=3),
    dict(text="Khác tập test và khác cách tính CS-WER nên bảng này không xếp hạng hai bên. V5B chốt trên DEV trước khi giải mã TEST.", size=14, bullet=True, after=0)])
notes(s, "Chi tiết và cảnh báo ở mục 2.5 của báo cáo; số trong same_metric_*.csv.")

# ================================================================== 10 V1 architecture
s = new_slide("Mô hình đề xuất V1: PhoWhisper-small + LoRA", section="Chương III · Mô hình đề xuất")
blocks = [("Audio 16 kHz", MINT, DARK), ("Log-mel 80 bin", MINT, DARK), ("Encoder\nPhoWhisper (đóng băng)\n+ LoRA r32", TEAL, WHITE),
          ("Decoder\nPhoWhisper (đóng băng)\n+ LoRA r32", TEAL, WHITE), ("Văn bản\n+ guard giải mã", MINT, DARK)]
bw, bg_ = 2.1, 0.4
for i, (t, c, tc) in enumerate(blocks):
    x = 0.6 + i * (bw + bg_)
    card(s, x, 1.8, bw, 1.5, c)
    textbox(s, x + 0.05, 1.8, bw - 0.1, 1.5, [dict(text=t, size=15, bold=True, color=tc, align=PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)
    if i < len(blocks) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.05), Inches(2.4), Inches(bg_ - 0.1), Inches(0.3))
        fill(a, CORAL)
textbox(s, 0.6, 3.7, 12.1, 3.2, [
    dict(text="Điểm cải tiến so với Baseline", size=20, bold=True, color=DARK, font=HEAD, after=8),
    dict(text="LoRA trên cả attention (q, k, v, out_proj) và FFN (fc1, fc2): 52 MB tham số huấn luyện thêm, backbone giữ nguyên.", size=16, bullet=True),
    dict(text="Huấn luyện trộn 4 nguồn (VIVOS, FLEURS, VietMed, ViMedCSS) để thích nghi miền y khoa mà vẫn giữ tiếng Việt chung.", size=16, bullet=True),
    dict(text="Nhãn huấn luyện chuẩn hóa cùng luật chấm điểm; guard giải mã chặn lặp từ và clip im lặng.", size=16, bullet=True),
    dict(text="Các bản sau: V3 chia lại dữ liệu theo video, V4 thêm SpecAugment, V5 đổi backbone sang medium.", size=16, bullet=True)])

# ================================================================== 11 V1 results
s = new_slide("Kết quả V1 và tín hiệu cải thiện", section="Chương III · Mô hình đề xuất")
b0, b1, v1 = devo.loc["B0"], devo.loc["B1"], devo.loc["V1"]
rows = [["DEV chính thức", "B0", "B1", "V1"],
        ["WER tổng %", fmt(b0.wer), fmt(b1.wer), fmt(v1.wer)],
        ["CER %", fmt(b0.cer), fmt(b1.cer), fmt(v1.cer)],
        ["Recall thuật ngữ %", fmt(b0.term_recall, 1), fmt(b1.term_recall, 1), fmt(v1.term_recall, 1)],
        ["ViMedCSS %", fmt(b0.wer_ViMedCSS), fmt(b1.wer_ViMedCSS), fmt(v1.wer_ViMedCSS)],
        ["VietMed %", fmt(b0.wer_VietMed), fmt(b1.wer_VietMed), fmt(v1.wer_VietMed)],
        ["FLEURS %", fmt(b0.wer_FLEURS), fmt(b1.wer_FLEURS), fmt(v1.wer_FLEURS)],
        ["VIVOS %", fmt(b0.wer_VIVOS), fmt(b1.wer_VIVOS), fmt(v1.wer_VIVOS)]]
table(s, 0.6, 1.5, 6.2, rows, [2.4, 1.2, 1.2, 1.2], size=14, row_h=0.52, bold_first_col=True, align_right=(1, 2, 3))
stat(s, 0.6, 5.85, 2.9, 1.1, "−6,25 điểm", "so với B0 (−25% tương đối)", color=GREEN, big_size=22)
stat(s, 3.7, 5.85, 3.1, 1.1, "13% → 71%", "recall thuật ngữ y khoa", color=GREEN, big_size=22)
image(s, "fig9b_curves_compact.png", 7.0, 1.45, 5.7, 3.1)
card(s, 7.0, 4.7, 5.7, 2.25, MINT)
textbox(s, 7.15, 4.75, 5.4, 2.2, [
    dict(text="Cải thiện và cái giá", size=16, bold=True, color=DARK, font=HEAD, after=3),
    dict(text="Giảm lớn ở ViMedCSS (−12,3) và VietMed (−4,1), đúng miền đích.", size=13, bullet=True, after=3),
    dict(text="VIVOS xấu đi (0,42 → 6,92) do lr 1e-3 làm huấn luyện phân kỳ (đường xám).", size=13, bullet=True, after=3),
    dict(text="Split chính thức rò rỉ video, nên phần ViMedCSS còn lạc quan.", size=13, bullet=True, after=3)])

# ================================================================== 12 failure analysis
s = new_slide("Phân tích lỗi: lỗi đến từ đâu", section="Chương III · Mô hình đề xuất")
image(s, "fig11_failure_categories.png", 0.5, 1.4, 7.4, 2.75)
stat(s, 8.2, 1.5, 2.2, 1.3, "69,3%", "lỗi nằm ở VietMed", color=CORAL, big_size=26)
stat(s, 10.5, 1.5, 2.2, 1.3, "52,7%", "lỗi do lệch biên clip", color=CORAL, big_size=26)
textbox(s, 8.2, 3.0, 4.5, 1.2, [dict(text="V5B trên DEV: 18.981 lỗi từ trên 4.978 clip; 579 clip (11,6%) đúng hoàn toàn.", size=14, color=MUTED)])
rows = [["Nguyên nhân", "Bằng chứng"],
        ["Nhãn không khớp phạm vi audio", "Kết quả dài hơn nhãn ở biên clip (suy luận từ văn bản): 52,7% lỗi, 68,3% lỗi VietMed"],
        ["Thuật ngữ hiếm, tên riêng", "phage 0/20, tarlov 0/16, wegener 0/9; chưa từng thấy: recall 20,8%"],
        ["Nhãn không nhất quán", "insulin → inulin (11 lần); nhãn Train có cả insulin (56) và inulin (167)"],
        ["Từ chức năng đồng âm", "các ↔ cái (111 lần), vữa → vỡ (47 lần)"],
        ["Số và vòng lặp beam", "“100 feet” so với “một trăm feet”; “nhnhnh…” ở clip FLEURS"]]
table(s, 0.6, 4.3, 12.1, rows, [3.2, 8.9], size=13, row_h=0.43, bold_first_col=True)

# ================================================================== 13 V2.. proposals
s = new_slide("Đề xuất cải tiến V2 → V5 dựa trên phân tích lỗi", section="Chương III · Mô hình đề xuất")
rows = [["Bản", "Thay đổi", "Lý do từ phân tích lỗi", "Kết quả"],
        ["V2", "V1 huấn luyện tiếp, lr 1e-4", "V1 chỉ khá khi lr giảm xuống", "19,08% (V1: 18,48%), không cải thiện"],
        ["V3", "Chia lại ViMedCSS theo video; lr 3e-4 từ model gốc", "Rò rỉ video 92,7%; lr 1e-3 phân kỳ", "18,26% trên DEV sạch (B0V: 24,83%)"],
        ["V3B", "Thêm beam search 5", "Giảm lỗi giải mã", "17,44% (−0,83)"],
        ["V4", "Thêm SpecAugment", "WER DEV-subset chạm đáy ở bước 1800", "17,96% (−0,30; KTC 95% −0,54 đến −0,05)"],
        ["V4B", "V4 + beam 5", "Cộng dồn hai kỹ thuật", "17,26% (−1,01 so với V3)"],
        ["V5", "Backbone PhoWhisper-medium, 2 epoch", "Lỗi còn lại là thuật ngữ hiếm và VietMed; model lớn học tốt hơn", "15,67% (−2,29 so với V4)"],
        ["V5B", "V5 + beam 5", "Như V3B, V4B", "15,33%, tốt nhất (TEST 15,16%)"]]
table(s, 0.6, 1.5, 12.1, rows, [0.8, 3.8, 4.2, 3.3], size=14, row_h=0.62, bold_first_col=True, highlight=(7,))
textbox(s, 0.6, 6.55, 12.1, 0.5, [dict(text="V5 đổi nhiều biến cùng lúc (backbone, lr, epoch), nên không quy toàn bộ mức giảm cho kích thước mô hình.", size=13, color=MUTED)])

# ================================================================== 14 ablation
s = new_slide("Bảng tổng hợp Ablation trên cùng DEV", section="Chương III · Mô hình đề xuất")
image(s, "fig6_ablation_dev_v3.png", 0.4, 1.4, 6.6, 3.2)
rows = [["", "greedy", "beam 5"],
        ["small, không SpecAugment", "V3 18,26", "V3B 17,44"],
        ["small, SpecAugment", "V4 17,96", "V4B 17,26"],
        ["medium, SpecAugment", "V5 15,67", "V5B 15,33"]]
table(s, 7.3, 1.5, 5.4, rows, [2.6, 1.4, 1.4], size=14, row_h=0.5, bold_first_col=True, highlight=(3,))
textbox(s, 7.3, 3.7, 5.4, 1.3, [
    dict(text="SpecAugment −0,30 và beam −0,83 cộng dồn gần độc lập (−1,01).", size=14, bullet=True),
    dict(text="Với medium, beam chỉ thêm 0,34 điểm (small: 0,70).", size=14, bullet=True)])
rows = [["DEV V3, 4.978 clip", "B0V", "V3", "V4B", "V5B"],
        ["WER %", fmt(dev3.loc["B0V"].wer), fmt(dev3.loc["V3"].wer), fmt(dev3.loc["V4B"].wer), fmt(dev3.loc["V5B"].wer)],
        ["Recall thuật ngữ %", fmt(dev3.loc["B0V"].term_recall, 1), fmt(dev3.loc["V3"].term_recall, 1), fmt(dev3.loc["V4B"].term_recall, 1), fmt(dev3.loc["V5B"].term_recall, 1)],
        ["ViMedCSS %", fmt(dev3.loc["B0V"].wer_ViMedCSS), fmt(dev3.loc["V3"].wer_ViMedCSS), fmt(dev3.loc["V4B"].wer_ViMedCSS), fmt(dev3.loc["V5B"].wer_ViMedCSS)],
        ["VietMed %", fmt(dev3.loc["B0V"].wer_VietMed), fmt(dev3.loc["V3"].wer_VietMed), fmt(dev3.loc["V4B"].wer_VietMed), fmt(dev3.loc["V5B"].wer_VietMed)]]
table(s, 0.6, 4.85, 12.1, rows, [3, 1.5, 1.5, 1.5, 1.5], size=14, row_h=0.42, bold_first_col=True, align_right=(1, 2, 3, 4))
notes(s, "Chọn mô hình chỉ trên DEV. TEST của split V3 đã được xem trong lúc phát triển nên không còn là số mù.")

# ================================================================== 14b O3W setup
off = pd.read_csv(TAB / "same_metric_official.csv")
obs = pd.read_csv(TAB / "official_test_by_source.csv").set_index("system")
freeze = json.loads((ROOT / "reports/final_freeze_official/freeze.json").read_text(encoding="utf-8"))
dev_tab = {r["decode"]: r for r in freeze["dev_table"]}
o = lambda sp, name: off[(off.split == sp) & (off.system == name)].iloc[0]
s = new_slide("O3W: train trên split chính thức để so trực tiếp với paper", section="Chương III · Mô hình đề xuất")
rows = [["Hạng mục", "O3W"],
        ["Backbone", "Whisper-large-v3 (1,55 tỉ tham số), đóng băng"],
        ["Fine-tune", "LoRA r32 / α64 như V1–V5, SpecAugment như V4"],
        ["Mới", "Oversampling thuật ngữ hiếm: 407 clip có thuật ngữ ≤ 5 lần trong Train, lặp 3 lần"],
        ["Dữ liệu", "Train chính thức 4 nguồn (29.864 clip sau oversampling); không dùng split theo video"],
        ["Huấn luyện", "Colab A100 80 GB, bf16, batch 32, lr 1e-4, 2 epoch, khoảng 2,1 giờ"],
        ["Chốt mô hình", "Chọn greedy hay beam 5 trên DEV chính thức, ghi freeze.json (sha256 adapter) trước khi giải mã TEST"]]
table(s, 0.6, 1.5, 7.3, rows, [1.7, 5.6], size=13, row_h=0.62, bold_first_col=True)
rows = [["DEV chính thức", "greedy", "beam 5"],
        ["WER tổng %", fmt(dev_tab["dev"]["wer"]), fmt(dev_tab["dev_beam5"]["wer"])],
        ["ViMedCSS %", fmt(dev_tab["dev"]["vimedcss"]), fmt(dev_tab["dev_beam5"]["vimedcss"])],
        ["Recall thuật ngữ %", fmt(dev_tab["dev"]["term_recall"], 1), fmt(dev_tab["dev_beam5"]["term_recall"], 1)]]
table(s, 8.2, 1.5, 4.5, rows, [2.1, 1.2, 1.2], size=14, row_h=0.5, bold_first_col=True, align_right=(1, 2), highlight=())
textbox(s, 8.2, 3.6, 4.5, 0.9, [dict(text=f"Chọn beam 5, chốt lúc {freeze['frozen_at'][11:16]} ngày 30/10, trước lần giải mã TEST/HARD đầu tiên.", size=13, color=MUTED)])
stat(s, 8.2, 4.55, 2.15, 1.45, "15,07%", "WER DEV chính thức (V1: 18,48%)", color=GREEN, big_size=24)
stat(s, 10.55, 4.55, 2.15, 1.45, "81,3%", "recall thuật ngữ DEV (V1: 71,3%)", color=GREEN, big_size=24)
textbox(s, 0.6, 6.05, 12.1, 0.9, [
    dict(text="O3W đổi nhiều biến cùng lúc so với V5 (backbone, split, oversampling, lr); ngân sách Colab chỉ đủ một lần train nên chưa có ablation riêng cho oversampling.", size=13, color=MUTED)])
notes(s, "Ngân sách khoảng 45 đơn vị Colab nên chỉ train O3W, không train O1/O2/O3P như kế hoạch đầy đủ.")

# ================================================================== 14c O3W vs papers
s = new_slide("O3W so với 3 paper trên cùng tập test", section="Chương III · Mô hình đề xuất")
image(s, "fig14_official_vs_paper.png", 0.4, 1.35, 7.6, 2.9)
rows = [["Tập test (paper)", "Paper tốt nhất", "Nhóm chạy lại", "O3W"],
        ["ViMedCSS TEST (ViMedCSS)", "23,67 (AG)", f"small {fmt(o('TEST', 'B0').wer)}", fmt(o("TEST", "O3W").wer)],
        ["ViMedCSS HARD (ViMedCSS)", "33,73 (AG)", f"small {fmt(o('HARD', 'B0').wer)}", fmt(o("HARD", "O3W").wer)],
        ["VietMed test (VietMed)", "29,6 (XLSR-53-Viet)", f"large {fmt(obs.loc['B2'].wer_VietMed)}", fmt(obs.loc["O3W"].wer_VietMed)],
        ["VIVOS test (PhoWhisper)", "4,67 (large)", f"large {fmt(obs.loc['B2'].wer_VIVOS)}", fmt(obs.loc["O3W"].wer_VIVOS)]]
table(s, 0.6, 4.4, 7.4, rows, [2.7, 1.9, 1.6, 1.2], size=13, row_h=0.44, bold_first_col=True, align_right=(1, 2, 3), highlight=(4,))
card(s, 8.3, 1.45, 4.4, 5.45, MINT)
textbox(s, 8.45, 1.55, 4.1, 5.3, [
    dict(text="Đọc kết quả", size=17, bold=True, color=DARK, font=HEAD, after=4),
    dict(text="Cách chấm của nhóm cho zero-shot thấp hơn paper 5–10 điểm, nên số O3W và số paper không so trực tiếp được. Muốn so phải chấm cùng transcript và cùng scorer.", size=13, bullet=True, after=4),
    dict(text=f"Thích nghi có tác dụng rõ: WER giảm {fmt(o('TEST', 'O3W').wer_red_vs_B0, 1)}% so với B0, cùng dữ liệu và cùng cách chấm.", size=13, bullet=True, after=4),
    dict(text=f"CS-WER giảm {fmt(o('TEST', 'O3W').cs_red_vs_B0, 1)}% so với B0; recall thuật ngữ {fmt(o('TEST', 'O3W').term_recall, 1)}% (khớp chuỗi con, cận trên; khớp nguyên từ 69,9%).", size=13, bullet=True, after=4),
    dict(text=f"Clip thuộc video không có trong Train: TEST {fmt(o('TEST', 'O3W').wer_unseen)} ({o('TEST', 'O3W').n_unseen} clip), HARD {fmt(o('HARD', 'O3W').wer_unseen)} ({o('HARD', 'O3W').n_unseen} clip). Chỉ là quan sát nội bộ, chưa đo được tác dụng của rò rỉ.", size=13, bullet=True, after=4),
    dict(text="VIVOS chưa đạt: PhoWhisper đã học VIVOS; Whisper-large-v3 thì không.", size=13, bullet=True, bold=True, color=CORAL, after=0)])
notes(s, "So sánh không cùng điều kiện: O3W 1,55 tỉ tham số và train thêm VIVOS/FLEURS/VietMed; AG của paper dùng PhoWhisper-small và chỉ ViMedCSS. Số trong tables/same_metric_official.csv.")

# ================================================================== 15 visual outputs
s = new_slide("Kết quả đầu ra tiêu biểu của V5B", section="Chương IV · Kết luận")
ex = pd.read_csv(TAB / "failure_examples.csv")
good = [("Đúng hoàn toàn (ViMedCSS)", "nếu các bạn muốn da môi của mình nó ẩm mượt hơn thì các bạn cũng có thể lấy một lượng vaseline",
         "nếu các bạn muốn da môi của mình nó ẩm mượt hơn thì các bạn cũng có thể lấy một lượng vaseline", GREEN),
        ("Sai thuật ngữ (ViMedCSS)", "đấy thì á hả nuôi cấy được vaccine", "thì à người kế được van xì một lần nữa", CORAL),
        ("Lệch biên clip (VietMed)", "giữ được thăng bằng hoặc là có rối loạn thị giác",
         "giữ được thăng bằng hoặc là có rối loạn về thị giác tuy nhiên thì thì những cái triệu chứng này nó ít gặp hơn", CORAL),
        ("Số viết thành chữ (FLEURS)", "nước tràn qua đê kéo rộng đến 100 feet", "nước tràn qua đê kéo gọng đến một trăm feet", CORAL)]
for i, (tag, ref, hyp, col) in enumerate(good):
    y = 1.5 + i * 1.3
    card(s, 0.6, y, 12.1, 1.18, GREY)
    textbox(s, 0.75, y + 0.05, 3.0, 1.1, [dict(text=tag, size=15, bold=True, color=col, font=HEAD)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 3.8, y + 0.05, 8.8, 1.1, [
        dict(runs=[dict(text="Nhãn:  ", bold=True, color=MUTED), dict(text=ref)], size=14, after=3),
        dict(runs=[dict(text="V5B:   ", bold=True, color=MUTED), dict(text=hyp)], size=14)], anchor=MSO_ANCHOR.MIDDLE)
textbox(s, 0.6, 6.75, 11, 0.4, [dict(text="Các câu lấy từ predictions.jsonl của V5B trên DEV; xem thêm tables/failure_examples.csv.", size=12, color=MUTED)])

# ================================================================== 16 feasibility
s = new_slide("Đánh giá tính khả thi xây dựng sản phẩm", section="Chương IV · Kết luận")
rows = [["Tiêu chí", "V4B (small)", "V5 (medium)", "V5B (medium, beam 5)", "O3W (large-v3, beam 5)*"],
        ["WER trên DEV", "17,26%", "15,67%", "15,33%", f"{fmt(dev_tab['dev_beam5']['wer'])}%"],
        ["WER trên TEST", "17,38%", "15,58%", "15,16%", f"{fmt(obs.loc['O3W'].wer_all)}%"],
        ["Recall thuật ngữ DEV", "66,4%", "69,7%", "72,5%", f"{fmt(dev_tab['dev_beam5']['term_recall'], 1)}%"],
        ["Tham số backbone", "244M", "769M", "769M", "1,55B"],
        ["Adapter LoRA", f"{lat.loc['V4B (small)'].adapter_MB:.0f} MB", f"{lat.loc['V5 (medium)'].adapter_MB:.0f} MB", f"{lat.loc['V5B (medium)'].adapter_MB:.0f} MB", "231 MB"],
        ["RTF giải mã", fmt(lat.loc['V4B (small)'].rtf, 3), fmt(lat.loc['V5 (medium)'].rtf, 3), fmt(lat.loc['V5B (medium)'].rtf, 3), "0,037 (A100)"],
        ["VRAM đỉnh khi suy luận", f"{fmt(lat.loc['V4B (small)'].eval_gpu_gb, 1)} GB", f"{fmt(lat.loc['V5 (medium)'].eval_gpu_gb, 1)} GB", f"{fmt(lat.loc['V5B (medium)'].eval_gpu_gb, 1)} GB", "chưa đo"]]
table(s, 0.6, 1.5, 7.5, rows, [2.0, 1.2, 1.2, 1.5, 1.6], size=12, row_h=0.45, bold_first_col=True, align_right=(1, 2, 3, 4))
textbox(s, 0.6, 5.15, 7.5, 0.3, [dict(text="* O3W đo trên DEV/TEST chính thức và A100; các bản V trên split chia theo video. Không so trực tiếp.", size=11, color=MUTED)])
card(s, 8.3, 1.5, 4.4, 2.0, DARK)
textbox(s, 8.45, 1.55, 4.1, 1.9, [
    dict(text="KẾT LUẬN", size=13, bold=True, color=MINT),
    dict(text="Khả thi có điều kiện", size=26, bold=True, color=WHITE, font=HEAD),
    dict(text="Đủ để đóng gói ở Task 3", size=14, color=MINT)], anchor=MSO_ANCHOR.MIDDLE)
textbox(s, 8.3, 3.65, 4.4, 1.0, [dict(text="RTF 0,18 – 0,32 nghĩa là nhanh hơn thời gian thực 3 – 6 lần trên GPU laptop 6 GB.", size=14, color=TEXT)])
textbox(s, 0.6, 5.5, 12.1, 1.5, [
    dict(text="Điều kiện cần xử lý trước khi phát hành", size=17, bold=True, color=DARK, font=HEAD),
    dict(text="Dùng được cho gõ nháp và tóm tắt ghi âm, chưa đủ cho văn bản y khoa không cần soát (O3W vẫn sót 20% thuật ngữ trên TEST, 50% trên HARD).", size=14, bullet=True),
    dict(text="Beam search gây lặp ở một số clip: cần cơ chế rơi về greedy. O3W cần GPU lớn hơn (backbone fp16 khoảng 3,1 GB); chưa đo trên laptop, CPU hay di động.", size=14, bullet=True)])

# ================================================================== 17 plan task 3
s = new_slide("Kế hoạch thực hiện Task 3", section="Chương IV · Kết luận")
plan = [("Chốt mô hình", "O3W và V5B đều đã chốt trên DEV rồi mới đo TEST. Còn: chọn hệ đóng gói (V5B nhẹ, O3W chính xác hơn)."),
        ("Cải thiện thuật ngữ", "Ablation O2 (không oversampling); PhoWhisper-large + LoRA để vượt VIVOS; từ điển y khoa, chuẩn hóa số; fallback greedy."),
        ("Đóng gói", "API suy luận (FastAPI) tải backbone fp16 + adapter LoRA; chạy GPU ≥ 4 GB; đo thêm CPU hoặc lượng tử hóa."),
        ("Giao diện và tích hợp", "UI tải audio và hiển thị văn bản kèm thuật ngữ; nối với nhóm MT qua đầu ra ASR đã chuẩn hóa."),
        ("Báo cáo", "Viết báo cáo tốt nghiệp từ tài liệu Task 2; thêm tập đánh giá nhãn sạch (nghe lại ~500 clip).")]
for i, (h, d) in enumerate(plan):
    y = 1.5 + i * 1.05
    c = card(s, 0.6, y + 0.1, 0.65, 0.65, TEAL, MSO_SHAPE.OVAL)
    textbox(s, 0.6, y + 0.1, 0.65, 0.65, [dict(text=str(i + 1), size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 1.5, y, 3.0, 0.85, [dict(text=h, size=18, bold=True, color=DARK, font=HEAD)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 4.6, y, 8.1, 0.85, [dict(text=d, size=15)], anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Phân công cụ thể từng thành viên cần điền theo nhóm.")

# ================================================================== 18 thanks
s = new_slide(None, dark=True)
textbox(s, 0.9, 2.6, 11.5, 1.5, [dict(text="Cảm ơn thầy cô và các bạn đã lắng nghe", size=40, bold=True, color=WHITE, font=HEAD)])
textbox(s, 0.9, 4.2, 11.5, 1.0, [dict(text="Hỏi đáp (Q&A)", size=24, color=MINT)])

out = OUT / "ASR_Task2_Slides.pptx"
prs.save(out)
print(out, state["n"], "slides")
