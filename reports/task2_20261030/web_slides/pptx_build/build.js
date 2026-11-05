// ASR + MT Task 2 deck as an editable PowerPoint (content ported from the web deck).
const path = require("path");
const pptxgen = require("pptxgenjs");
const D = require("./data.json");

const OUT = process.argv[2] || "deck_raw.pptx";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Nhận dạng và dịch giọng nói y khoa Việt–Anh";
pres.author = "Nhóm 2";
const FONT = "Calibri";
const MONO = "Consolas";

const K = {
  scrub: "0A3A42", scrub2: "0F4E57", scrub3: "1A6771", paper: "E4EEEB", paper2: "CFE0DC", sheet: "F3F8F6",
  ink: "08262C", ink2: "3B5A5E", on: "E8F3F0", on2: "AFCFCB",
  asr: "FFB21A", asrS: "CF8700", asrT: "FFD27A", asrInk: "8F5200",
  mt: "3F80F2", mtS: "2858B8", mtT: "86AEF7", mtInk: "1B4BAE",
  n: "7E9C9B", nS: "5C7877", nT: "A9C2C0", harm: "BC1F45", harmT: "FFB3C2",
};
pres.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: "vi-VN" };

/* ---------- layouts ---------- */
const X0 = 0.667, CW = 12.0, X1 = X0 + CW;
function master(name, bg, fg, mut, rule, hasHead) {
  const objects = [];
  if (hasHead) {
    objects.push({ placeholder: { options: { name: "kicker", type: "body", x: X0, y: 0.34, w: CW, h: 0.3, fontFace: FONT, fontSize: 11, color: mut, bold: false, margin: 0, valign: "top", align: "left" }, text: "" } });
    objects.push({ placeholder: { options: { name: "title", type: "title", x: X0, y: 0.62, w: CW, h: 0.72, fontFace: FONT, fontSize: 28, bold: true, color: fg, margin: 0, valign: "middle", align: "left" }, text: "" } });
    objects.push({ line: { x: X0, y: 1.45, w: CW, h: 0, line: { color: rule, width: 0.75 } } });
    objects.push({ text: { text: "Nhóm 2 · ASR & MT", options: { x: X0, y: 7.05, w: 4, h: 0.25, fontFace: FONT, fontSize: 9, color: mut, margin: 0 } } });
  }
  pres.defineSlideMaster({ title: name, background: { color: bg }, objects, slideNumber: hasHead ? { x: 11.67, y: 7.05, w: 1, h: 0.25, fontFace: FONT, fontSize: 9, color: mut, align: "right" } : undefined });
}
master("LIGHT", K.paper, K.ink, K.ink2, K.paper2, true);
master("DARK", K.scrub, K.on, K.on2, K.scrub3, true);
pres.defineSlideMaster({
  title: "COVER", background: { color: K.scrub },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 1.35, w: 9.4, h: 1.75, fontFace: FONT, fontSize: 46, bold: true, color: K.on, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: X0, y: 3.2, w: 9, h: 0.5, fontFace: FONT, fontSize: 18, color: K.on2, margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "THANKS", background: { color: K.scrub },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.9, w: 11, h: 1.0, fontFace: FONT, fontSize: 54, bold: true, color: K.on, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: X0, y: 2.1, w: 9, h: 0.5, fontFace: FONT, fontSize: 20, color: K.on2, margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
});

/* ---------- helpers ---------- */
const SH = () => ({ type: "outer", color: "000000", opacity: 0.2, blur: 7, offset: 3, angle: 45 });
function rich(str, o = {}) {
  const out = [];
  const re = /<(mark|ins|b)>(.*?)<\/\1>/g;
  let i = 0, m;
  while ((m = re.exec(str))) {
    if (m.index > i) out.push({ text: str.slice(i, m.index), options: { ...o } });
    const t = m[2];
    if (m[1] === "mark") out.push({ text: t, options: { ...o, bold: true, color: "8A1230", highlight: K.harmT } });
    else if (m[1] === "ins") out.push({ text: t, options: { ...o, bold: true, color: o.dark ? K.asr : K.asrInk } });
    else out.push({ text: t, options: { ...o, bold: true } });
    i = m.index + m[0].length;
  }
  if (i < str.length) out.push({ text: str.slice(i), options: { ...o } });
  out.forEach((r) => delete r.options.dark);
  return out;
}
function txt(s, text, o = {}) {
  const runs = typeof text === "string" ? rich(text, { fontFace: o.font || FONT, fontSize: o.fs || 13, color: o.color || K.ink, bold: !!o.bold, dark: o.dark }) : text;
  s.addText(runs, { x: o.x, y: o.y, w: o.w, h: o.h || 0.4, margin: 0, isTextBox: true, valign: o.valign || "top", align: o.align || "left", fit: "none", paraSpaceAfter: o.psa || 0, lineSpacingMultiple: o.lh || 1.05, ...(o.charSpacing ? { charSpacing: o.charSpacing } : {}) });
}
function head(s, kick, title) {
  s.addText(kick, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
}
function tile(s, { x, y, w, h, em, b, t, dark, bfs = 14, tfs = 12 }) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: dark ? K.scrub2 : K.sheet }, line: { type: "none" }, shadow: SH(), objectName: "Thẻ" });
  const col = dark ? K.on : K.ink, mut = dark ? K.on2 : K.ink2;
  const runs = [];
  if (em) runs.push({ text: em, options: { fontSize: 9.5, color: mut, fontFace: FONT, breakLine: true } });
  if (b) runs.push({ text: b, options: { fontSize: bfs, bold: true, color: col, fontFace: FONT, breakLine: !!t } });
  if (t) runs.push(...rich(t, { fontSize: tfs, color: col, fontFace: FONT, dark }));
  s.addText(runs, { x: x + 0.16, y: y + 0.1, w: w - 0.32, h: h - 0.2, margin: 0, isTextBox: true, valign: "middle", fit: "none", paraSpaceAfter: 2 });
}
function note(s, { x, y, w, h, title, text, fs = 12, dark }) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: dark ? K.scrub2 : K.scrub }, line: { type: "none" }, shadow: SH(), objectName: "Ghi chú" });
  const runs = [];
  if (title) runs.push({ text: title, options: { bold: true, fontSize: fs + 3, color: "FFFFFF", fontFace: FONT, breakLine: true } });
  runs.push(...rich(text, { fontSize: fs, color: K.on, fontFace: FONT, dark: true }));
  s.addText(runs, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h - 0.24, margin: 0, isTextBox: true, valign: "middle", fit: "none", paraSpaceAfter: 4 });
}
function stat(s, { x, y, w, num, label, fs = 30, dark, color }) {
  txt(s, num, { x, y, w, h: fs / 72 * 1.25, fs, bold: true, color: color || (dark ? K.on : K.scrub2), valign: "bottom" });
  txt(s, label, { x, y: y + fs / 72 * 1.3, w, h: 0.55, fs: 10.5, color: dark ? K.on2 : K.ink2 });
}
function bullets(s, { x, y, w, h, items, fs = 12.5, dark }) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const r = rich(it, { fontSize: fs, color: dark ? K.on : K.ink, fontFace: FONT, dark });
    r[0].options.bullet = { indent: 14 };
    r[r.length - 1].options.breakLine = !last;
    runs.push(...r);
  });
  s.addText(runs, { x, y, w, h, margin: 0, isTextBox: true, valign: "top", fit: "none", paraSpaceAfter: 6 });
}
const CUBE = pres.shapes.CUBE || "cube";
function cube(s, { x, y, w, h, fill, text, tc, fs = 12, sub, subfs = 9.5 }) {
  const runs = [];
  if (text) runs.push({ text, options: { bold: true, fontSize: fs, color: tc, fontFace: FONT, breakLine: !!sub } });
  if (sub) runs.push({ text: sub, options: { fontSize: subfs, color: tc, fontFace: FONT } });
  s.addText(runs, { shape: CUBE, x, y, w, h, fill: { color: fill }, line: { type: "none" }, margin: [2, 4, 2, 4], align: "center", valign: "middle", fit: "none", shadow: SH() });
}
function arrow(s, x, y, w, color) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2, endArrowType: "triangle" } });
}
function archRow(s, bl, x0, baseY, gap, ac, fs, subfs) {
  let cx = x0;
  const pos = bl.map((b) => { const x = cx; cx += b[2] + gap; return x; });
  bl.forEach((b, i) => {
    for (let k = b[6]; k >= 1; k--) s.addShape(CUBE, { x: pos[i] - 0.07 * k, y: baseY - b[3], w: b[2], h: b[3], fill: { color: b[4] }, line: { color: K.scrub, width: 0.75 } });
    cube(s, { x: pos[i], y: baseY - b[3], w: b[2], h: b[3], fill: b[4], tc: b[5], text: b[0], sub: b[1], fs, subfs });
  });
  bl.forEach((b, i) => {
    if (i < bl.length - 1) { const a = pos[i] + b[2] + 0.05, e = pos[i + 1] - 0.07 * bl[i + 1][6] - 0.05; arrow(s, a, baseY - 0.75, e - a, ac); }
  });
}
function slab(s, { x, y, w, cols, rows, fs = 12.5, rh = 0.5, hh = 0.45, hi = {}, hm = {}, hdFs = 11 }) {
  const tot = cols.reduce((a, c) => a + (c.w || 1), 0);
  const colW = cols.map((c) => (w * (c.w || 1)) / tot);
  const bd = [{ type: "none" }, { type: "none" }, { pt: 0.75, color: K.paper2 }, { type: "none" }];
  const head = cols.map((c) => ({ text: c.t, options: { bold: true, color: "FFFFFF", fill: { color: K.scrub }, align: c.r ? "right" : "left", valign: "middle", fontSize: hdFs, fontFace: FONT, border: [{ type: "none" }, { type: "none" }, { type: "none" }, { type: "none" }] } }));
  const body = rows.map((r, i) =>
    r.map((v, j) => {
      const h = hi[i], m = hm[i];
      const base = { fontSize: fs, fontFace: FONT, color: m ? "FFFFFF" : K.ink, bold: !!(h || m) };
      return { text: typeof v === "string" ? rich(v, base) : v, options: { fill: { color: h ? K.asr : m ? K.mt : i % 2 ? "EAF2EF" : K.sheet }, align: cols[j].r ? "right" : "left", valign: "middle", fontSize: fs, fontFace: FONT, color: base.color, bold: base.bold, border: bd } };
    })
  );
  const H = hh + rows.reduce((a, _, i) => a + (Array.isArray(rh) ? rh[i] : rh), 0);
  s.addShape(pres.shapes.RECTANGLE, { x: x + 0.09, y: y + 0.09, w, h: H, fill: { color: "9DB8B3" }, line: { type: "none" }, shadow: { type: "outer", color: "000000", opacity: 0.22, blur: 9, offset: 5, angle: 45 }, objectName: "Độ dày bảng" });
  s.addTable([head, ...body], { x, y, w, colW, rowH: [hh, ...rows.map((_, i) => (Array.isArray(rh) ? rh[i] : rh))], margin: [2, 8, 2, 8], fontFace: FONT, objectName: "Bảng" });
  return H;
}
const BC = { n: K.n, d: K.scrub2, a: K.asr };
function bars3d(s, { x, y, h, items, max, bw = 0.6, gap = 0.1, fs = 12, lfs = 10, floorW }) {
  let cx = x;
  const yb = y + h;
  const end = items.reduce((a, it) => a + (it.sp ? it.sp : bw + gap), x);
  s.addShape(pres.shapes.RECTANGLE, { x: x - 0.12, y: yb, w: (floorW || end - x) + 0.2, h: 0.07, fill: { color: K.paper2 }, line: { type: "none" } });
  for (const it of items) {
    if (it.sp) { cx += it.sp; continue; }
    const H = Math.max(0.09, (it.v / max) * h);
    s.addShape(CUBE, { x: cx, y: yb - H, w: bw, h: H, fill: { color: BC[it.c || "n"] }, line: { type: "none" }, shadow: SH(), objectName: "Cột " + it.l });
    txt(s, it.t, { x: cx - 0.25, y: yb - H - 0.36, w: bw + 0.5, h: 0.3, fs, bold: true, align: "center", valign: "bottom" });
    txt(s, it.l, { x: cx - 0.2, y: yb + 0.1, w: bw + 0.4, h: 0.4, fs: lfs, color: K.ink2, align: "center" });
    cx += bw + gap;
  }
}
function hbars(s, { x, y, rows, max, lw = 1.9, tw = 3.0, step = 0.36 }) {
  rows.forEach((r, i) => {
    const yy = y + i * step, p = (r.v / max) * tw;
    txt(s, r.l, { x, y: yy, w: lw - 0.1, h: 0.3, fs: 11.5, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: x + lw + 0.04, y: yy + 0.05, w: p, h: 0.22, fill: { color: "8FB0AB" }, line: { type: "none" } });
    s.addShape(pres.shapes.RECTANGLE, { x: x + lw, y: yy + 0.02, w: p, h: 0.22, fill: { color: r.a ? K.asr : K.scrub2 }, line: { type: "none" } });
    txt(s, r.t, { x: x + lw + p + 0.12, y: yy, w: 0.7, h: 0.28, fs: 11, bold: true, valign: "middle" });
  });
}
const P = (s, text, o) => txt(s, text, { fs: 14, ...o });
const sec = (name) => pres.addSection({ title: name });
const L = (s) => ({ masterName: "LIGHT", sectionTitle: s });
const Dk = (s) => ({ masterName: "DARK", sectionTitle: s });
const ch = { a: "I · Dữ liệu", b: "II · Baseline ASR", c: "III · Phương pháp đề xuất ASR", d: "IV · Dịch máy và ghép hệ thống", e: "V · Kết luận" };
const SEC = { 0: "Mở đầu", a: "I · Dữ liệu", b: "II · Baseline ASR", c: "III · Phương pháp đề xuất", d: "IV · Dịch máy", e: "V · Kết luận" };
Object.values(SEC).forEach(sec);

/* ---------- 1 cover ---------- */
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: SEC[0] });
  s.addImage({ path: "terrain.png", x: 0, y: 0, w: 13.333, h: 7.5, altText: "Phổ log-mel của một clip ViMedCSS dựng thành địa hình" });
  txt(s, "Báo cáo tiến độ · Task 2", { x: X0, y: 0.85, w: 8, h: 0.3, fs: 14, bold: true, color: K.asr, charSpacing: 1 });
  s.addText("Nhận dạng và dịch giọng nói y khoa Việt–Anh", { placeholder: "title" });
  s.addText("Dữ liệu, thực nghiệm và kết quả mô hình", { placeholder: "sub" });
  txt(s, "Phổ log-mel · ViMedCSS TEST “…điều trị bệnh lý thalassemia ở Việt Nam…”", { x: 6.9, y: 6.2, w: 5.77, h: 0.3, fs: 10, color: K.on2, align: "right" });
  txt(s, [{ text: "Nhóm 2", options: { fontSize: 13, color: K.on, fontFace: FONT, breakLine: true } }, { text: "Nguyen Luong Hai Dang · Tran Trung Tin · Vo Hieu Thang", options: { fontSize: 13, color: K.on, fontFace: FONT } }], { x: X0, y: 6.55, w: 8, h: 0.55 });
  txt(s, "GVHD: Nguyen Xuan Huy", { x: 8.67, y: 6.8, w: 4, h: 0.3, fs: 13, color: K.on, align: "right" });
}

/* ---------- 2 agenda ---------- */
{
  const s = pres.addSlide(L(SEC[0]));
  head(s, "Nội dung báo cáo", "Các phần trình bày");
  const rows = [["I", "Xử lý và phân tích dữ liệu", "4–7"], ["II", "Thực nghiệm hai mô hình baseline", "8–10"], ["III", "Phát triển mô hình: V1 → V7", "11–19"], ["IV", "Dịch máy và tích hợp ASR + MT", "20–21"], ["V", "Đầu ra và hướng triển khai", "22–24"]];
  rows.forEach((r, i) => {
    const y = 1.85 + i * 1.0;
    txt(s, r[0], { x: X0, y, w: 0.9, h: 0.75, fs: 28, bold: true, color: K.scrub3, valign: "middle" });
    txt(s, r[1], { x: 1.75, y, w: 8.5, h: 0.75, fs: 21, valign: "middle" });
    txt(s, r[2], { x: 10.4, y, w: 2.267, h: 0.75, fs: 14, color: K.ink2, align: "right", valign: "middle", font: MONO });
    s.addShape(pres.shapes.LINE, { x: X0, y: y + 0.85, w: CW, h: 0, line: { color: K.paper2, width: 0.75 } });
  });
}

/* ---------- 3 overview ---------- */
{
  const s = pres.addSlide(Dk(SEC[0]));
  head(s, "Tổng quan hệ thống", "Hệ thống nhận dạng và dịch giọng nói y khoa");
  const bl = [
    ["Audio", "16 kHz, mono, tối đa 30 s", 1.55, K.scrub3, K.on],
    ["Whisper + LoRA", "chép lời, beam 5", 2.0, K.asr, "1D1400"],
    ["Văn bản Việt–Anh", "chuẩn hóa, số thành chữ", 2.0, K.n, K.ink],
    ["mBART + LoRA", "dịch, beam 5", 1.9, K.mt, "FFFFFF"],
    ["English", "bản dịch", 1.55, K.scrub3, K.on],
  ];
  const gap = 0.55, tot = bl.reduce((a, b) => a + b[2], 0) + gap * 4;
  let cx = X0 + (CW - tot) / 2;
  bl.forEach((b, i) => {
    cube(s, { x: cx, y: 2.2, w: b[2], h: 2.0, fill: b[3], tc: b[4], text: b[0], sub: b[1], fs: 15, subfs: 10.5 });
    if (i < bl.length - 1) arrow(s, cx + b[2] + 0.08, 3.3, gap - 0.16, K.on2);
    cx += b[2] + gap;
  });
  txt(s, [{ text: "Nhận dạng giọng nói (ASR). ", options: { bold: true, color: K.asr, fontSize: 15, fontFace: FONT } }, { text: "4 nguồn dữ liệu, 2 baseline, 7 phiên bản V1 → V7, đã chấm trên TEST và HARD.", options: { color: K.on, fontSize: 15, fontFace: FONT } }], { x: X0, y: 5.2, w: 5.8, h: 1.2 });
  txt(s, [{ text: "Dịch máy (MT). ", options: { bold: true, color: K.mtT, fontSize: 15, fontFace: FONT } }, { text: "Đã tích hợp với ASR; đang bổ sung đánh giá trên tập dữ liệu chung.", options: { color: K.on, fontSize: 15, fontFace: FONT } }], { x: 6.87, y: 5.2, w: 5.8, h: 1.2 });
}

/* ---------- 4 data ---------- */
{
  const s = pres.addSlide(L(SEC.a));
  head(s, ch.a, "Dữ liệu: 4 bộ ngữ liệu, 40.914 clip");
  slab(s, {
    x: X0, y: 1.95, w: 7.7, fs: 14, rh: 0.55, hh: 0.5, hi: { 3: 1 },
    cols: [{ t: "Nguồn", w: 1.15 }, { t: "Loại", w: 0.8 }, { t: "Train", r: 1, w: 0.85 }, { t: "Dev", r: 1, w: 0.75 }, { t: "Test", r: 1, w: 0.75 }, { t: "Giờ", r: 1, w: 0.65 }, { t: "Dung lượng", r: 1, w: 1.05 }],
    rows: [["VIVOS", "Online", "11.476", "184", "760", "15,7", "915 MB"], ["FLEURS", "Online", "2.994", "361", "857", "13,3", "627 MB"], ["VietMed", "Online", "2.773", "2.912", "3.437", "15,8", "699 MB"], ["ViMedCSS", "Online", "11.832", "1.714", "1.614", "30,8", "1,9 GB"]],
  });
  txt(s, "ViMedCSS có thêm 658 clip HARD. Nhóm không thu dữ liệu Offline. Định dạng sau xử lý: FLAC 16 kHz mono trong parquet.", { x: X0, y: 5.0, w: 7.7, h: 0.6, fs: 12.5, color: K.ink2 });
  txt(s, "Trường dữ liệu: id, source, split, group_id, speaker_id, text, cs_terms, duration, audio.", { x: X0, y: 5.75, w: 7.7, h: 0.5, fs: 12.5, color: K.ink2 });
  txt(s, "Ví dụ bản chép lời", { x: 8.75, y: 1.65, w: 3.9, h: 0.35, fs: 16, bold: true });
  [["VIVOS · câu đọc", "có thể lành lặn mà không để lại vết sẹo"], ["FLEURS · câu đọc", "Bản tin truyền hình cho thấy có khói trắng bốc ra từ nhà máy."], ["VietMed · thoại y khoa", "việc cho con bú mẹ giúp giảm nguy cơ bị bệnh tiểu đường tuýp hai ở mẹ về lâu dài"], ["ViMedCSS · code-switching", "Những loại <ins>virus</ins> có hại không chỉ gây ảnh hưởng không nhỏ tới sức khỏe…"]].forEach((t, i) => tile(s, { x: 8.75, y: 2.1 + i * 1.12, w: 3.92, h: 1.0, em: t[0], t: t[1], tfs: 12 }));
}

/* ---------- 5 cleaning ---------- */
{
  const s = pres.addSlide(L(SEC.a));
  head(s, ch.a, "Quy trình làm sạch dữ liệu");
  const T = [
    ["Bước 1", "Audio", "16 kHz, mono, FLAC; bỏ clip ngoài (0, 30 s]"], ["Bước 2", "Văn bản", "NFC, chữ thường, bỏ dấu câu, thống nhất khi huấn luyện và đánh giá"], ["Bước 3", "Trùng lặp", "Loại <b>4</b> clip trùng ở ViMedCSS"], ["Bước 4", "Nhiễu", "Loại <b>64</b> clip im lặng, đỉnh biên độ dưới 0,001"],
    ["Bước 5", "Nhãn bất thường", "Loại <b>14</b> clip nhãn dày hơn 16 token/giây khỏi Train"], ["Bước 6", "Giá trị thiếu", "Giữ trống speaker_id ở FLEURS và ViMedCSS"], ["Bước 7", "Rò rỉ", "Chia lại ViMedCSS theo video, 0 video trùng"], ["Bước 8", "Tập TEST", "Chỉ giải mã khi bật cờ --allow-test"],
  ];
  T.forEach((t, i) => tile(s, { x: X0 + (i % 4) * 2.2, y: 1.8 + Math.floor(i / 4) * 2.45, w: 2.05, h: 2.3, em: t[0], b: t[1], t: t[2], bfs: 15, tfs: 12 }));
  note(s, { x: 9.55, y: 1.8, w: 3.117, h: 3.3, title: "Gán nhãn", text: "Transcript lấy từ chính corpus. Nhãn thuật ngữ cs_terms do tác giả ViMedCSS cung cấp. Nhóm không gán nhãn thủ công, nên chưa có Kappa.", fs: 12 });
  txt(s, "Chất lượng nhãn chưa được đánh giá độc lập. Dự kiến hai người nghe lại 300–500 clip để kiểm tra các trường hợp lệch giữa nhãn và âm thanh.", { x: 9.55, y: 5.3, w: 3.117, h: 1.3, fs: 11.5, color: K.ink2 });
}

/* ---------- 6 distribution ---------- */
{
  const s = pres.addSlide(L(SEC.a));
  head(s, ch.a, "Phân bố dữ liệu giữa Train và DEV");
  bars3d(s, {
    x: 0.95, y: 1.95, h: 3.3, max: 58, bw: 0.6, gap: 0.08, lfs: 9.5, fs: 12,
    items: [{ l: "VIVOS", v: 23.4, t: "23,4", c: "n" }, { l: "DEV", v: 3.3, t: "3,3", c: "a" }, { sp: 0.35 }, { l: "FLEURS", v: 13.5, t: "13,5", c: "n" }, { l: "DEV", v: 8.4, t: "8,4", c: "a" }, { sp: 0.35 }, { l: "VietMed", v: 10.6, t: "10,6", c: "n" }, { l: "DEV", v: 55.6, t: "55,6", c: "a" }, { sp: 0.35 }, { l: "ViMedCSS", v: 52.6, t: "52,6", c: "n" }, { l: "DEV", v: 32.7, t: "32,7", c: "a" }],
  });
  txt(s, "% số từ theo nguồn. Xám là Train, vàng là DEV. ViMedCSS và thống kê thuật ngữ theo bản chia theo video; Train chính thức có 610 thuật ngữ.", { x: X0, y: 6.2, w: 7.3, h: 0.7, fs: 11.5, color: K.ink2 });
  slab(s, { x: 8.4, y: 1.85, w: 4.27, fs: 13, rh: 0.46, hh: 0.44, cols: [{ t: "Split", w: 1 }, { t: "Clip", r: 1 }, { t: "% clip", r: 1 }], rows: [["Train", "29.075", "71,1%"], ["Dev", "5.171", "12,6%"], ["Test", "6.668", "16,3%"]] });
  txt(s, "Split chính thức của 4 nguồn, gần tỷ lệ 70/15/15.", { x: 8.4, y: 3.95, w: 4.27, h: 0.3, fs: 11.5, color: K.ink2 });
  [["22,5%", "top-10 thuật ngữ trên tổng số lượt"], ["173", "thuật ngữ chỉ xuất hiện 1 lần"], ["390", "thuật ngữ dưới 5 lần, 47% của 831"], ["20,8%", "recall thuật ngữ chưa thấy, V6"]].forEach((t, i) => stat(s, { x: 8.4 + (i % 2) * 2.2, y: 4.45 + Math.floor(i / 2) * 1.2, w: 2.0, num: t[0], label: t[1], fs: 30 }));
}

/* ---------- 7 imbalance ---------- */
{
  const s = pres.addSlide(L(SEC.a));
  head(s, ch.a, "Chia lại dữ liệu và tăng mẫu thuật ngữ hiếm");
  txt(s, "Rò rỉ video", { x: X0, y: 1.7, w: 3.5, h: 0.35, fs: 17, bold: true });
  txt(s, "% clip DEV thuộc video có trong Train", { x: X0, y: 2.1, w: 3.5, h: 0.3, fs: 11.5, color: K.ink2 });
  bars3d(s, { x: 1.0, y: 2.85, h: 2.4, max: 100, bw: 0.95, gap: 0.85, lfs: 11, fs: 13, items: [{ l: "Trước", v: 92.7, t: "92,7%", c: "n" }, { l: "Sau", v: 0.6, t: "0%", c: "a" }], floorW: 3.0 });
  txt(s, "Chia lại ViMedCSS theo video, 80/10/10, seed 13.", { x: X0, y: 5.85, w: 3.4, h: 0.6, fs: 12.5 });
  txt(s, "Thuật ngữ hiếm", { x: 4.5, y: 1.7, w: 3.5, h: 0.35, fs: 17, bold: true });
  txt(s, "% lượt thuật ngữ xuất hiện tối đa 5 lần", { x: 4.5, y: 2.1, w: 3.5, h: 0.3, fs: 11.5, color: K.ink2 });
  bars3d(s, { x: 4.85, y: 2.85, h: 2.4, max: 10.3, bw: 0.95, gap: 0.85, lfs: 11, fs: 13, items: [{ l: "Trước", v: 3.35, t: "3,35%", c: "n" }, { l: "Sau", v: 9.35, t: "9,35%", c: "a" }], floorW: 3.0 });
  txt(s, "Oversampling: 407 clip × 3, Train 29.050 → 29.864 clip.", { x: 4.5, y: 5.85, w: 3.5, h: 0.6, fs: 12.5 });
  slab(s, { x: 8.4, y: 1.85, w: 4.27, fs: 13, rh: 0.5, hh: 0.45, hi: { 0: 1, 1: 1 }, cols: [{ t: "Số lần xuất hiện", w: 1.25 }, { t: "Trước", r: 1 }, { t: "Sau", r: 1 }], rows: [["1", "120", "360"], ["2–5", "292", "876"], ["6–20", "2.792", "2.836"], ["21–100", "5.425", "5.469"], ["> 100", "3.665", "3.679"]] });
  txt(s, "SpecAugment 5% + 5%: DEV 18,26 → 17,96.", { x: 8.4, y: 5.2, w: 4.27, h: 0.35, fs: 12.5 });
  txt(s, "Chưa dùng class weighting, focal loss. Tác dụng riêng của oversampling chưa đo.", { x: 8.4, y: 5.65, w: 4.27, h: 0.7, fs: 12, color: K.ink2 });
}

/* ---------- 8 environment ---------- */
{
  const s = pres.addSlide(L(SEC.b));
  head(s, ch.b, "Môi trường thực nghiệm");
  tile(s, { x: X0, y: 1.8, w: 2.35, h: 1.2, em: "Laptop", b: "RTX 4050 6 GB", t: "V1 → V6, 2 baseline, fp16" });
  tile(s, { x: X0 + 2.55, y: 1.8, w: 2.35, h: 1.2, em: "Colab", b: "A100 80 GB", t: "V7 và các lần chấm lại, bf16" });
  txt(s, "<b>Cố định:</b> seed 13, batch hiệu dụng 32, AdamW mặc định của Trainer, LoRA r32, α64, dropout 0,05.", { x: X0, y: 3.2, w: 4.9, h: 0.75, fs: 12 });
  txt(s, "<b>Framework:</b> PyTorch 2.6.0, Transformers 4.46.3, PEFT 0.13.2.", { x: X0, y: 4.0, w: 4.9, h: 0.5, fs: 12 });
  note(s, { x: X0, y: 4.65, w: 4.9, h: 1.6, text: "2 baseline chạy zero-shot nên không có learning rate, optimizer, epochs. Điểm đồng nhất: cùng dữ liệu, cùng luật chuẩn hóa, cùng cách giải mã greedy.", fs: 12 });
  slab(s, { x: 5.95, y: 1.85, w: 6.72, fs: 13.5, rh: 0.56, hh: 0.5, hi: { 6: 1 }, cols: [{ t: "Phiên bản", w: 2.3 }, { t: "Backbone", w: 1.7 }, { t: "lr", r: 1, w: 0.8 }, { t: "Epoch", r: 1, w: 0.7 }], rows: [["V1 LoRA cơ bản", "PhoWhisper-small", "1e-3", "3"], ["V2 Train tiếp", "PhoWhisper-small", "1e-4", "2"], ["V3 Chia theo video", "PhoWhisper-small", "3e-4", "3"], ["V4 SpecAugment", "PhoWhisper-small", "3e-4", "3"], ["V5 SpecAugment + Beam", "PhoWhisper-small", "3e-4", "3"], ["V6 Medium + Beam", "PhoWhisper-medium", "2e-4", "2"], ["V7 Large + Thuật ngữ hiếm", "Whisper-large-v3", "1e-4", "2"]] });
  txt(s, "V5 dùng lại mô hình của V4, chỉ đổi cách giải mã sang beam 5.", { x: 5.95, y: 6.4, w: 6.72, h: 0.35, fs: 12, color: K.ink2 });
}

/* ---------- 9 baseline vs paper ---------- */
{
  const s = pres.addSlide(L(SEC.b));
  head(s, ch.b, "Baseline zero-shot: đối chiếu kết quả công bố");
  slab(s, { x: X0, y: 1.9, w: CW, fs: 15.5, rh: 0.78, hh: 0.55, hi: { 1: 1 }, hdFs: 12, cols: [{ t: "Mô hình", w: 1.6 }, { t: "Tập đánh giá", w: 1.5 }, { t: "WER paper", r: 1 }, { t: "WER nhóm", r: 1 }, { t: "Ghi chú", w: 1.8 }], rows: [["PhoWhisper-small", "ViMedCSS TEST", "36,31", "28,02", "cùng tập test, khác scorer"], ["PhoWhisper-small", "VIVOS test", "6,33", "6,28", "tái lập gần đúng"], ["Whisper-small", "ViMedCSS", "50,03", "46,45", "paper TEST, nhóm DEV"], ["Whisper-small", "FLEURS vi", "chưa tra được", "25,61", "nhóm DEV, thiếu số paper"]] });
  ["Chênh 0,05 điểm trên VIVOS cho thấy pipeline đánh giá hợp lý.", "Whisper-small lặp từ và ảo giác trên VietMed: 73,92% không guard, 55,23% có guard.", "Baseline 2 chưa giải mã TEST, nên dòng 3 và 4 không so trực tiếp với paper."].forEach((t, i) => txt(s, t, { x: X0 + i * 4.1, y: 5.95, w: 3.8, h: 1.0, fs: 14.5 }));
}

/* ---------- 10 gap ---------- */
{
  const s = pres.addSlide(L(SEC.b));
  head(s, ch.b, "Chênh lệch WER so với kết quả công bố");
  slab(s, { x: X0, y: 1.9, w: 7.6, fs: 14, rh: 0.9, hh: 0.65, hdFs: 11.5, cols: [{ t: "Zero-shot", w: 1.7 }, { t: "TEST paper", r: 1 }, { t: "TEST nhóm", r: 1 }, { t: "Lệch", r: 1, w: 0.7 }, { t: "HARD paper", r: 1 }, { t: "HARD nhóm", r: 1 }, { t: "Lệch", r: 1, w: 0.7 }], rows: [["PhoWhisper-small", "36,31", "28,02", "8,3", "46,27", "36,04", "10,2"], ["PhoWhisper-large", "31,24", "23,14", "8,1", "37,37", "27,14", "10,2"], ["Whisper-large-v3", "34,47", "29,61", "4,9", "39,35", "33,14", "6,2"]] });
  txt(s, "Cùng mô hình, cùng tập test, không huấn luyện. Cần kiểm tra bản chép lời tham chiếu, quy tắc chuẩn hóa và cách tính WER để giải thích chênh lệch.", { x: X0, y: 5.6, w: 7.6, h: 0.7, fs: 12.5, color: K.ink2 });
  txt(s, "Các baseline đều chạy zero-shot. Khác biệt về dữ liệu huấn luyện chỉ áp dụng khi so sánh mô hình đề xuất.", { x: X0, y: 6.35, w: 7.6, h: 0.6, fs: 12.5, color: K.ink2 });
  [["Cần đối chiếu", "Tiền xử lý", "Bài báo chưa mô tả quy tắc chuẩn hóa; ảnh hưởng đến WER cần được kiểm tra."], ["Khác biệt ở mô hình đề xuất", "Dữ liệu huấn luyện", "Mô hình đề xuất dùng thêm VIVOS, FLEURS, VietMed; bài báo chỉ dùng ViMedCSS."], ["Cần thống nhất cách tính", "Chỉ số CS-WER", "Bài báo chưa nêu công thức. Cách tính hiện tại cho 95,19, so với 62,55 được công bố."]].forEach((t, i) => tile(s, { x: 8.65, y: 1.85 + i * 1.6, w: 4.02, h: 1.45, em: t[0], b: t[1], t: t[2], bfs: 14.5, tfs: 12 }));
}

/* ---------- 11 architecture V1 ---------- */
{
  const s = pres.addSlide(Dk(SEC.c));
  head(s, ch.c, "Kiến trúc V1: PhoWhisper-small + LoRA");
  const bl = [["Log-Mel", "80 × 3.000", 0.9, 1.5, K.n, K.ink, 0], ["Conv1D × 2", "GELU, stride 2", 1.15, 1.5, K.n, K.ink, 0], ["Encoder × 12", "Self-Attention, FFN", 1.55, 2.3, K.scrub3, K.on, 2], ["Decoder × 12", "Masked Attention, Cross-Attention, FFN", 1.75, 2.3, K.asr, "1D1400", 2], ["Token", "beam, greedy", 0.9, 1.5, K.n, K.ink, 0]];
  const baseY = 5.1;
  archRow(s, bl, X0 + 0.15, baseY, 0.5, K.on2, 11.5, 9);
  s.addShape(pres.shapes.RECTANGLE, { x: X0, y: baseY + 0.02, w: 8.0, h: 0.06, fill: { color: K.scrub3 }, line: { type: "none" } });
  tile(s, { x: 9.0, y: 1.75, w: 3.667, h: 1.5, dark: true, em: "Một lớp LoRA", b: "y = W·x + 2·B·A·x", t: "W đóng băng; A: d → 32, B: 32 → d; α/r = 64/32 = 2; dropout 0,05.", bfs: 16, tfs: 11.5 });
  bullets(s, { x: 9.0, y: 3.5, w: 3.667, h: 1.9, dark: true, fs: 12, items: ["Gắn vào q, k, v, out_proj và fc1, fc2 của cả encoder, decoder.", "d_model 768, 12 đầu attention, FFN 3.072.", "12,98 triệu trên 254,71 triệu tham số."] });
  txt(s, [{ text: "So với baseline: ", options: { bold: true, color: K.asr, fontSize: 12, fontFace: FONT } }, { text: "thêm LoRA, train trộn 4 nguồn, nhãn chuẩn hóa cùng luật chấm.", options: { color: K.on, fontSize: 12, fontFace: FONT } }], { x: 9.0, y: 5.5, w: 3.667, h: 0.9 });
}

/* ---------- 12 V1 result ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "V1: WER giảm 6,25 điểm trên DEV");
  slab(s, { x: X0, y: 1.8, w: 6.1, fs: 13, rh: 0.46, hh: 0.44, hi: { 0: 1 }, cols: [{ t: "DEV chính thức", w: 1.35 }, { t: "Baseline 1", r: 1 }, { t: "Baseline 2", r: 1 }, { t: "V1", r: 1, w: 0.75 }], rows: [["WER %", "24,73", "58,75", "18,48"], ["CER %", "19,25", "46,35", "14,36"], ["Recall thuật ngữ", "13,0%", "16,1%", "71,3%"], ["ViMedCSS", "28,36", "46,46", "16,10"], ["VietMed", "25,87", "73,92", "21,80"], ["FLEURS", "10,89", "25,61", "11,12"], ["VIVOS", "0,42", "20,45", "6,92"]] });
  txt(s, "<b>Kết quả:</b> ViMedCSS giảm 12,3 điểm, thuật ngữ tăng từ 13,0% lên 71,3%.", { x: X0, y: 6.1, w: 6.1, h: 0.6, fs: 12.5 });
  // WER curves on DEV-subset (V1 clipped at 40, as in the web deck); V5 key = medium run, shown as V6
  s.addShape(pres.shapes.RECTANGLE, { x: 7.1, y: 1.8, w: 5.57, h: 3.75, fill: { color: K.sheet }, line: { type: "none" }, shadow: SH() });
  const ks = ["V1", "V3", "V4", "V5"], nm = { V1: "V1", V3: "V3", V4: "V4", V5: "V6" };
  const X = [...new Set(ks.flatMap((k) => D.curves[k].map((a) => a[0])))].sort((a, b) => a - b);
  const ser = ks.map((k) => { const m = new Map(D.curves[k].map((a) => [a[0], Math.min(a[1], 40)])); return { name: nm[k], values: X.map((x) => (m.has(x) ? m.get(x) : null)) }; });
  s.addChart(pres.charts.SCATTER, [{ name: "Bước huấn luyện", values: X }, ...ser], {
    x: 7.2, y: 1.9, w: 5.37, h: 3.55, chartColors: [K.harm, K.scrub2, K.n, K.asrS], lineSize: 2.25, lineDataSymbol: "none",
    valAxisMinVal: 0, valAxisMaxVal: 40, valAxisMajorUnit: 10, catAxisMinVal: 0, catAxisMaxVal: 3000, catAxisMajorUnit: 1000,
    valAxisLabelFontFace: FONT, catAxisLabelFontFace: FONT, valAxisLabelFontSize: 10, catAxisLabelFontSize: 10, valAxisLabelColor: K.ink2, catAxisLabelColor: K.ink2,
    valGridLine: { color: K.paper2, size: 0.75 }, catGridLine: { style: "none" }, showValAxisTitle: true, valAxisTitle: "WER % (DEV-subset)", valAxisTitleFontSize: 10, valAxisTitleColor: K.ink2, valAxisTitleFontFace: FONT,
    showCatAxisTitle: true, catAxisTitle: "Bước huấn luyện", catAxisTitleFontSize: 10, catAxisTitleColor: K.ink2, catAxisTitleFontFace: FONT,
    showLegend: true, legendPos: "t", legendFontSize: 11, legendFontFace: FONT, legendColor: K.ink,
  });
  txt(s, "<b>Hạn chế:</b> lr 1e-3 làm huấn luyện phân kỳ quanh bước 700, loss 0,4 → 4,6 (V1 vọt tới 320%, trục cắt ở 40). V3 trở đi dùng lr thấp hơn. Các cặp từ nhầm lẫn được trình bày ở phần phân tích lỗi.", { x: 7.1, y: 5.75, w: 5.57, h: 1.1, fs: 11.5 });
}

/* ---------- 13 error analysis ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "Phân tích lỗi nhận dạng");
  txt(s, "Phân loại 18.981 lỗi từ (%)", { x: X0, y: 1.7, w: 5.6, h: 0.35, fs: 15, bold: true });
  hbars(s, { x: X0, y: 2.15, max: 60, lw: 1.9, tw: 3.1, rows: [{ l: "Lệch biên clip", v: 52.7, t: "52,7", a: 1 }, { l: "Thay thế rải rác", v: 34.8, t: "34,8" }, { l: "Chèn, lặp", v: 6.6, t: "6,6" }, { l: "Xóa đoạn", v: 5.9, t: "5,9" }] });
  txt(s, "Các cặp từ thường nhầm (số lần)", { x: X0, y: 3.85, w: 5.6, h: 0.35, fs: 15, bold: true });
  hbars(s, { x: X0, y: 4.3, max: 80, lw: 1.9, tw: 3.1, rows: D.subs.slice(0, 6).map((r) => ({ l: r[0] + " → " + r[1], v: r[2], t: String(r[2]) })) });
  [["Giả thuyết, chưa nghe lại audio", "Nhãn dừng sớm hơn lời nói", "Nhãn: “…lấy một lượng vaseline”. Mô hình nghe tiếp: “chúng ta bôi lên môi sau đó…”"], ["V6, DEV", "Thuật ngữ hiếm, tên riêng", "phage 0/20, tarlov 0/16, wegener 0/9. Recall: 20,8% với thuật ngữ chưa gặp, 75,8% với thuật ngữ đã gặp khi huấn luyện."], ["Biến thể chính tả", "Nhãn không nhất quán", "Train có insulin 56 lần, inulin 167 lần, nên insulin → inulin bị tính sai 11 lần."]].forEach((t, i) => tile(s, { x: 6.9, y: 1.75 + i * 1.5, w: 5.77, h: 1.38, em: t[0], b: t[1], t: t[2], bfs: 14.5, tfs: 11.5 }));
  txt(s, "VietMed chiếm 69,3% số lỗi; lỗi ở biên clip chiếm 52,7%. Thống kê trên V6. V1 có dạng lỗi tương tự và thêm hiện tượng phân kỳ khi huấn luyện.", { x: 6.9, y: 6.3, w: 5.77, h: 0.65, fs: 11, color: K.ink2 });
}

/* ---------- 14 versions ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "Các phiên bản mô hình: V1 → V7");
  slab(s, { x: X0, y: 1.8, w: CW, fs: 12.5, rh: 0.58, hh: 0.45, hi: { 6: 1 }, cols: [{ t: "Phiên bản", w: 1.75 }, { t: "Vấn đề cần xử lý", w: 1.75 }, { t: "Thay đổi", w: 2.2 }, { t: "WER DEV", r: 1, w: 0.7 }, { t: "Nhận xét", w: 1.6 }], rows: [["V1 LoRA cơ bản", "Baseline sót thuật ngữ y khoa", "Thêm LoRA, train 4 nguồn", "18,48", "lr 1e-3 phân kỳ"], ["V2 Train tiếp", "V1 cải thiện khi giảm lr", "Train tiếp V1, lr 1e-4", "19,08", "không cải thiện"], ["V3 Chia theo video", "Rò rỉ 92,7%, lr phân kỳ", "Chia theo video, lr 3e-4", "18,26", "không trùng video giữa các tập"], ["V4 SpecAugment", "V3 quá khớp từ bước 1800", "Thêm SpecAugment", "17,96", "−0,30 [−0,54; −0,05]"], ["V5 SpecAugment + Beam", "Lỗi giải mã", "V4 giải mã beam 5", "17,26", "tốt nhất nhóm small"], ["V6 Medium + Beam", "Thuật ngữ hiếm, VietMed", "PhoWhisper-medium, beam 5", "15,33", "WER giảm ở cả 4 nguồn"], ["V7 Large + Thuật ngữ hiếm", "Cần chấm trên test của paper", "large-v3, oversampling, Train chính thức", "15,07", "mô hình cuối, beam 5"]] });
  txt(s, "V1, V2, V7 chấm trên DEV chính thức, V3 đến V6 trên DEV chia theo video, nên hai nhóm không so chéo. V6 và V7 đổi nhiều biến cùng lúc.", { x: X0, y: 6.45, w: CW, h: 0.5, fs: 11.5, color: K.ink2 });
}

/* ---------- 15 ablation ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "Ảnh hưởng của từng bước cải tiến");
  bars3d(s, { x: 0.95, y: 1.95, h: 3.7, max: 27, bw: 0.95, gap: 0.5, lfs: 11.5, fs: 14, items: [{ l: "Baseline 1", v: 24.83, t: "24,83", c: "n" }, { l: "V3", v: 18.26, t: "18,26", c: "d" }, { l: "V4", v: 17.96, t: "17,96", c: "d" }, { l: "V5", v: 17.26, t: "17,26", c: "d" }, { l: "V6", v: 15.33, t: "15,33", c: "a" }] });
  txt(s, "DEV chia theo video, 4.978 clip.", { x: X0, y: 6.3, w: 7, h: 0.35, fs: 11.5, color: K.ink2 });
  slab(s, { x: 8.3, y: 1.85, w: 4.37, fs: 13, rh: 0.5, hh: 0.45, hi: { 4: 1 }, cols: [{ t: "DEV chính thức", w: 1.7 }, { t: "WER", r: 1 }], rows: [["Baseline 1", "24,73"], ["Baseline 2", "58,75"], ["V1", "18,48"], ["V2", "19,08"], ["V7", "15,07"]] });
  txt(s, "ViMedCSS TEST theo video: <ins>30,02 → 14,78</ins>. V6 được chọn theo kết quả trên DEV.", { x: 8.3, y: 5.15, w: 4.37, h: 0.9, fs: 12.5 });
}

/* ---------- 16 V7 setup ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "V7: huấn luyện và đánh giá theo bộ chia chính thức");
  P(s, "V3–V6 dùng bộ chia lại theo video, có chứa clip từ TEST chính thức trong tập Train. Vì vậy, các phiên bản này không dùng để đối chiếu trên TEST của bài báo.", { x: X0, y: 1.85, w: 4.9, h: 1.6, fs: 13.5 });
  P(s, "V7 chỉ dùng Train chính thức của 4 nguồn, rồi giải mã trên TEST, HARD của ViMedCSS, VIVOS test và VietMed test.", { x: X0, y: 3.5, w: 4.9, h: 1.1, fs: 13.5 });
  note(s, { x: X0, y: 4.75, w: 4.9, h: 1.6, text: "Chọn cách giải mã trên DEV: beam 5 đạt WER 15,07%, so với 15,52% của greedy. Cấu hình được giữ nguyên khi đánh giá TEST và HARD.", fs: 12 });
  slab(s, { x: 5.95, y: 1.9, w: 6.72, fs: 13, rh: [0.55, 0.55, 0.55, 0.75, 0.55], hh: 0.45, hi: { 2: 1 }, cols: [{ t: "Hạng mục", w: 1 }, { t: "V7 Large + Thuật ngữ hiếm", w: 3 }], rows: [["Backbone", "Whisper-large-v3, 1,55 tỉ tham số, đóng băng"], ["Fine-tune", "LoRA r32, α64, dropout 0,05; SpecAugment 5% + 5%"], ["Mới", "Oversampling thuật ngữ hiếm: 407 clip × 3"], ["Huấn luyện", "A100, bf16, batch 32, lr 1e-4, 2 epoch, 1.868 bước, khoảng 2,1 giờ"], ["Giải mã", "Bỏ clip im lặng, tối đa 12 token/giây + 20, beam 5"]] });
}

/* ---------- 17 V7 results ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "Kết quả V7 trên TEST và HARD");
  const mk = (x, title, it) => {
    txt(s, title, { x, y: 1.7, w: 4, h: 0.35, fs: 16, bold: true });
    bars3d(s, { x: x + 0.2, y: 2.5, h: 3.0, max: 38, bw: 0.62, gap: 0.32, lfs: 10, fs: 12, items: it });
  };
  mk(X0, "TEST, 1.614 clip", [{ l: "Pho-small", v: 28.02, t: "28,02", c: "n" }, { l: "Pho-large", v: 23.14, t: "23,14", c: "n" }, { l: "Large-v3", v: 29.61, t: "29,61", c: "n" }, { l: "V7", v: 11.4, t: "11,40", c: "a" }]);
  mk(4.85, "HARD, 658 clip", [{ l: "Pho-small", v: 36.04, t: "36,04", c: "n" }, { l: "Pho-large", v: 27.14, t: "27,14", c: "n" }, { l: "Large-v3", v: 33.14, t: "33,14", c: "n" }, { l: "V7", v: 16.42, t: "16,42", c: "a" }]);
  stat(s, { x: 9.2, y: 1.85, w: 3.5, num: "69,9–79,9%", label: "recall thuật ngữ TEST, khớp nguyên từ đến chuỗi con", fs: 28 });
  stat(s, { x: 9.2, y: 3.45, w: 3.5, num: "45,8–50,3%", label: "recall thuật ngữ HARD", fs: 28 });
  txt(s, "249/338 thuật ngữ HARD không có trong Train của cả 4 nguồn. Cột xám: PhoWhisper-small, PhoWhisper-large, Whisper-large-v3 chưa huấn luyện. WER % trên ViMedCSS.", { x: 9.2, y: 4.8, w: 3.47, h: 1.6, fs: 11, color: K.ink2 });
}

/* ---------- 18 vs papers ---------- */
{
  const s = pres.addSlide(L(SEC.c));
  head(s, ch.c, "Đối chiếu với các nghiên cứu đã công bố");
  slab(s, { x: X0, y: 1.85, w: 7.7, fs: 13, rh: 0.62, hh: 0.45, cols: [{ t: "Tập test", w: 1.3 }, { t: "Hệ của paper", w: 1.5 }, { t: "Paper", r: 1, w: 0.7 }, { t: "V7", r: 1, w: 0.7 }, { t: "Số của V7", w: 1.2 }], rows: [["ViMedCSS TEST", "Attention Guide", "23,67", "11,40", "thấp hơn"], ["ViMedCSS HARD", "Attention Guide", "33,73", "16,42", "thấp hơn"], ["VietMed", "XLSR-53-Viet", "29,6", "20,43", "thấp hơn"], ["VIVOS", "PhoWhisper-large", "4,67", "7,03", "chưa đạt"]] });
  txt(s, "<ins>So sánh trong cùng thiết lập:</ins> cùng Whisper-large-v3, cùng tập test, cùng cách chấm, WER giảm từ 29,61 trước huấn luyện xuống 11,40 ở V7.", { x: X0, y: 5.0, w: 7.7, h: 1.2, fs: 14 });
  txt(s, "Khác biệt khi so sánh", { x: 8.7, y: 1.7, w: 3.97, h: 0.35, fs: 16, bold: true });
  [["Khác scorer", "Zero-shot của nhóm thấp hơn paper 5–10 điểm trên cùng mô hình."], ["Khác hệ thống", "V7 lớn gấp khoảng 6 lần PhoWhisper-small, train thêm 3 nguồn."], ["Thiếu đầu ra của AG", "Nhóm không có transcript đầu ra của hệ AG."]].forEach((t, i) => tile(s, { x: 8.7, y: 2.15 + i * 1.4, w: 3.97, h: 1.28, b: t[0], t: t[1], bfs: 14, tfs: 12 }));
}

/* ---------- 19 video leak pilot ---------- */
{
  const s = pres.addSlide(Dk(SEC.c));
  head(s, ch.c, "Ảnh hưởng của trùng video giữa Train và DEV");
  P(s, "Hai bộ Train cùng 29.913 clip, khác nhau ở 722 clip: bản sạch lấy từ video khác, bản rò rỉ lấy từ chính các video trong tập chấm.", { x: X0, y: 1.8, w: 5.3, h: 1.3, fs: 13.5, color: K.on });
  P(s, "Chấm chung trên <ins>probe_seen</ins> (809 clip, 192 video) và <ins>probe_unseen</ins> (1.409 clip, 179 video, đối chứng).", { x: X0, y: 3.2, w: 5.3, h: 1.1, fs: 13.5, color: K.on, dark: true });
  note(s, { x: X0, y: 4.55, w: 5.3, h: 1.5, dark: true, text: "Kết quả sơ bộ trên <b>1/3 seed</b>, với PhoWhisper-small + LoRA. Recall được bổ sung sau khi xem dữ liệu và dùng để phân tích thăm dò.", fs: 12 });
  txt(s, "DiD và khoảng tin cậy 95% theo video, điểm %", { x: 6.4, y: 1.7, w: 6.27, h: 0.35, fs: 15, bold: true, color: K.on });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.4, y: 2.15, w: 6.27, h: 3.3, fill: { color: K.scrub2 }, line: { type: "none" }, shadow: { type: "outer", color: "000000", opacity: 0.35, blur: 12, offset: 6, angle: 45 }, objectName: "Khung biểu đồ DiD" });
  const X = 6.75, Wd = 5.55, sc = (v) => X + ((v + 2) / 20) * Wd;
  [0, 5, 10, 15].forEach((v) => {
    s.addShape(pres.shapes.LINE, { x: sc(v), y: 2.35, w: 0, h: 2.65, line: { color: v ? "3A7A83" : K.on, width: v ? 1 : 2.25 } });
    txt(s, v ? "+" + v : "0", { x: sc(v) - 0.4, y: 5.05, w: 0.8, h: 0.28, fs: 11, color: K.on2, align: "center", font: MONO });
  });
  [["WER", 0.3, -0.63, 1.14, K.n, "+0,30", 3.0], ["Recall, khớp nguyên từ", 9.53, 4.39, 14.69, K.asr, "+9,53", 3.85], ["Recall, khớp chuỗi con", 10.91, 6.25, 15.83, K.asr, "+10,91", 4.6]].forEach((r, i) => {
    const [lab, pt, lo, hi, col, val, y] = r;
    s.addShape(pres.shapes.RECTANGLE, { x: sc(lo) + 0.05, y: y + 0.04, w: sc(hi) - sc(lo), h: 0.17, fill: { color: "04262C" }, line: { type: "none" } });
    s.addShape(pres.shapes.RECTANGLE, { x: sc(lo), y, w: sc(hi) - sc(lo), h: 0.17, fill: { color: col }, line: { type: "none" }, objectName: "Khoảng tin cậy " + lab });
    s.addShape(pres.shapes.OVAL, { x: sc(pt) - 0.11, y: y - 0.03, w: 0.23, h: 0.23, fill: { color: K.on }, line: { color: K.scrub, width: 2 } });
    if (i === 0) txt(s, lab + ": " + val, { x: sc(hi) + 0.2, y: y - 0.07, w: 2.6, h: 0.32, fs: 12, color: K.on });
    else txt(s, lab + ": " + val, { x: sc(lo), y: y - 0.4, w: 3.2, h: 0.32, fs: 12, color: K.on });
  });
  txt(s, "DiD là mức giảm ở seen trừ mức giảm ở unseen. <ins>WER:</ins> khoảng tin cậy chứa 0, chưa có bằng chứng về tác dụng. <ins>Recall:</ins> 10 thuật ngữ đóng góp 63/87 lượt tăng.", { x: 6.4, y: 5.7, w: 6.27, h: 1.1, fs: 12, color: K.on, dark: true });
}

/* ---------- 20 MT ---------- */
{
  const s = pres.addSlide(Dk(SEC.d));
  head(s, ch.d, "Dịch máy: mBART + LoRA");
  const bl = [["Tiền xử lý", "số thành chữ, chữ thường", 1.55, 1.5, K.n, K.ink, 0], ["Encoder × 12", "d_model 1.024", 1.55, 1.9, K.scrub3, K.on, 2], ["Decoder × 12", "16 đầu attention", 1.55, 1.9, K.mt, "FFFFFF", 2], ["English", "beam 5, 256 token", 1.35, 1.5, K.scrub3, K.on, 0]];
  const baseY = 4.25;
  archRow(s, bl, X0 + 0.15, baseY, 0.55, K.on2, 12, 9.5);
  s.addShape(pres.shapes.RECTANGLE, { x: X0, y: baseY + 0.02, w: 7.7, h: 0.06, fill: { color: K.scrub3 }, line: { type: "none" } });
  txt(s, [{ text: "Backbone ", options: { bold: true, color: K.mtT, fontSize: 12.5, fontFace: FONT } }, { text: "vinai/vinai-translate-vi2en-v2. ", options: { color: K.on, fontSize: 12.5, fontFace: FONT } }, { text: "Adapter ", options: { bold: true, color: K.mtT, fontSize: 12.5, fontFace: FONT } }, { text: "M4_s42: LoRA r16, α32, dropout 0,05, 8,65 triệu tham số trên q, k, v, out_proj, fc1, fc2.", options: { color: K.on, fontSize: 12.5, fontFace: FONT } }], { x: X0, y: 5.0, w: 7.7, h: 1.0 });
  txt(s, "Chạy thử trên 6 câu · RTX 4050", { x: 8.9, y: 1.75, w: 3.77, h: 0.35, fs: 15, bold: true, color: K.on });
  [["0,98 GB", "VRAM đỉnh, RTX 4050"], ["~0,5 s", "mỗi câu, beam 5"], ["34,7 MB", "adapter"], ["16,6 s", "tải mô hình"]].forEach((t, i) => stat(s, { x: 8.9 + (i % 2) * 1.95, y: 2.25 + Math.floor(i / 2) * 1.3, w: 1.85, num: t[0], label: t[1], fs: 26, dark: true }));
  note(s, { x: 8.9, y: 5.0, w: 3.77, h: 1.25, dark: true, text: "<b>Đang bổ sung:</b> dữ liệu và cấu hình huấn luyện MT; kết quả BLEU, chrF++, COMET, TER.", fs: 12 });
}

/* ---------- 21 ASR errors into MT ---------- */
{
  const s = pres.addSlide(Dk(SEC.d));
  head(s, ch.d, "Ảnh hưởng của lỗi ASR đến bản dịch");
  const m = D.mt[2];
  const T = [
    ["Người nói", "Nhưng trong một số ca nặng thì corticosteron có thể giúp <ins>giảm phù nề dây thần kinh</ins>."],
    ["ASR (V7) · sai cụm cuối câu", "nhưng trong một số ca nặng thì corticosteron có thể giúp <mark>giảm phụ nữ</mark>"],
    ["MT dịch từ câu đúng", m.en_ref.replace("reduce nerve edema", "<ins>reduce nerve edema</ins>")],
    ["MT dịch từ đầu ra ASR", m.en_hyp.replace("relieve women", "<mark>relieve women</mark>")],
  ];
  T.forEach((t, i) => tile(s, { x: X0, y: 1.8 + i * 1.12, w: 7.2, h: 1.0, dark: true, em: t[0], t: t[1], tfs: 12.5 }));
  txt(s, "Một câu ví dụ, không đại diện cho tỷ lệ lỗi.", { x: X0, y: 6.35, w: 7.2, h: 0.3, fs: 11.5, color: K.on2 });
  txt(s, "Tích hợp ASR và MT", { x: 8.2, y: 1.7, w: 4.47, h: 0.35, fs: 16, bold: true, color: K.on });
  bullets(s, { x: 8.2, y: 2.15, w: 4.47, h: 2.5, dark: true, fs: 12.5, items: ["ASR xuất chữ thường, không dấu câu. MT dùng đúng luật chuẩn hóa đó, thêm bước đọc số thành chữ.", "Gọi mt.translate(text) khi ASR nhận dạng xong một câu.", "Thư viện yêu cầu trùng phiên bản với môi trường ASR."] });
  txt(s, "Đánh giá tiếp theo", { x: 8.2, y: 4.7, w: 4.47, h: 0.35, fs: 16, bold: true, color: K.on });
  bullets(s, { x: 8.2, y: 5.15, w: 4.47, h: 1.6, dark: true, fs: 12.5, items: ["Điểm dịch khi đầu vào là câu đúng và khi là đầu ra ASR, trên cùng tập.", "VRAM khi ASR và MT chạy chung trên GPU 6 GB."] });
}

/* ---------- 22 end-to-end output ---------- */
{
  const s = pres.addSlide(L(SEC.e));
  head(s, ch.e, "Kết quả đầu ra: từ âm thanh đến bản dịch");
  const R = (arr) => arr;
  slab(s, { x: X0, y: 1.75, w: CW, fs: 11.5, rh: 0.74, hh: 0.42, cols: [{ t: "Người nói", w: 1.25 }, { t: "ASR (V7)", w: 1.25 }, { t: "MT dịch từ đầu ra ASR", w: 1.45 }], rows: R([
    ["…dao động từ 30 đến 300 mg, nhưng trung bình nằm vào khoảng 90 đến 100 mg.", "…dao động từ 30 đến 300 mg nhưng trung bình nằm vào khoảng 90 đến 100 mg", D.mt[0].en_hyp],
    ["…bệnh Wilson gây tổn thương gan đầu tiên ở trẻ nhỏ và trẻ dưới 13 tuổi.", "…bệnh wilson gây tổn thương gan đầu tiên ở trẻ nhỏ và trẻ dưới <mark>10</mark> tuổi", D.mt[3].en_hyp.replace("age of 10", "age of <mark>10</mark>")],
    ["…rối loạn hắc sắc tố melanin.", "…rối loạn hắc sắc tố <mark>[mất “melanin”]</mark>", D.mt[4].en_hyp],
    ["…bệnh u hạt gọi là Wegener này…có cái viêm mạch do hoại tử.", "…bệnh u <mark>hạm</mark> gọi là <mark>wienkenner</mark> này…có cái viêm mạch <mark>sau bạc mạch</mark> này", D.mt[5].en_hyp.replace("\"Wienkenner's lymphoma,\"", "<mark>\"Wienkenner's lymphoma,\"</mark>")],
  ]) });
  [["201", "khớp hoàn toàn"], ["636", "đúng thuật ngữ, sai từ khác"], ["107", "khớp một phần thuật ngữ"], ["215", "mất thuật ngữ"]].forEach((t, i) => stat(s, { x: X0 + i * 3.0, y: 5.5, w: 2.8, num: t[0], label: t[1], fs: 34 }));
  txt(s, "Thống kê trên 1.159 clip có thuật ngữ trong ViMedCSS TEST, dài 3,5–9 giây. Các câu minh họa được lấy từ mẫu ngẫu nhiên.", { x: X0, y: 6.65, w: CW, h: 0.3, fs: 10.5, color: K.ink2 });
}

/* ---------- 23 deployment ---------- */
{
  const s = pres.addSlide(L(SEC.e));
  head(s, ch.e, "Hướng triển khai hệ thống");
  slab(s, { x: X0, y: 1.85, w: 7.9, fs: 13.5, rh: 0.66, hh: 0.5, hi: { 1: 1 }, hm: { 3: 1 }, cols: [{ t: "Thành phần", w: 2 }, { t: "WER DEV", r: 1 }, { t: "Tốc độ", r: 1, w: 1.3 }, { t: "VRAM", r: 1 }, { t: "Adapter", r: 1 }], rows: [["V5 SpecAugment + Beam", "17,26", "RTF 0,208", "2,1 GB", "52 MB"], ["V6 Medium + Beam", "15,33", "RTF 0,321", "3,8 GB", "139 MB"], ["V7 Large + Thuật ngữ hiếm", "15,07*", "RTF 0,037 (A100)", "chưa đo", "231 MB"], ["Dịch máy MT", "chưa có", "~0,5 s/câu", "0,98 GB", "34,7 MB"]] });
  txt(s, "RTF là thời gian giải mã chia thời lượng audio, trên RTX 4050 trừ V7. *DEV chính thức, khác DEV của V5, V6. RTF không gồm các bước khác và chưa chứng minh chạy streaming.", { x: X0, y: 5.15, w: 7.9, h: 0.8, fs: 12, color: K.ink2 });
  txt(s, "<b>Triển khai:</b> web có backend GPU, xử lý từng câu. Đề xuất V6 cho máy 6 GB, V7 khi có GPU từ 8 GB và đã đo lại.", { x: X0, y: 6.05, w: 7.9, h: 0.8, fs: 14 });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.0, y: 1.85, w: 3.667, h: 0.95, fill: { color: "FFE3A6" }, line: { type: "none" }, shadow: SH() });
  txt(s, "Bản nháp có người kiểm tra", { x: 9.2, y: 1.9, w: 3.3, h: 0.85, fs: 17, bold: true, valign: "middle" });
  bullets(s, { x: 9.0, y: 3.1, w: 3.667, h: 3.7, fs: 13, items: ["Cần kiểm tra lại thuật ngữ trước khi sử dụng bản dịch. Hệ thống còn bỏ sót khoảng 3/10 thuật ngữ trên TEST.", "Đánh giá chất lượng dịch trên tập chung.", "Đo VRAM khi ASR và MT chạy đồng thời.", "Chuyển sang greedy khi beam lặp từ: V6 có 9 clip bị cắt trên TEST."] });
}

/* ---------- 24 thanks ---------- */
{
  const s = pres.addSlide({ masterName: "THANKS", sectionTitle: SEC.e });
  s.addImage({ path: "terrain.png", x: 0, y: 0, w: 13.333, h: 7.5, altText: "Phổ log-mel dựng thành địa hình" });
  s.addText("Cảm ơn thầy cô và các bạn", { placeholder: "title" });
  s.addText("Trao đổi và thảo luận", { placeholder: "sub" });
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log("written", OUT);
});
