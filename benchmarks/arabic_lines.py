"""Line-level OCR benchmark for Arabic / mixed text (synthetic).

    # 1) generate line images + ground truth
    python benchmarks/arabic_lines.py gen OUT_DIR --fonts DIR [--n 40] [--seed 1]
    # 2) run a recogniser (with the python of the venv that has it installed)
    python benchmarks/arabic_lines.py run OUT_DIR --engine paddle_ar --models-dir M --pred P.jsonl
    # 3) score one or more prediction files
    python benchmarks/arabic_lines.py score OUT_DIR P1.jsonl [P2.jsonl ...]

Each line has a category (ar_text, ar_indic_number, ar_with_latin_value, latin, ar_date)
and the exact value that matters (number / ID / name). Metrics:
  value_ok  - the value appears exactly in the output (after Arabic letter-form and
              digit normalisation; the decimal separator must be present)
  cer       - character error rate on the whole line (order-sensitive; mixed-direction
              lines are penalised for run-order differences)
Synthetic: fonts are whatever is in --fonts (only for evaluation, not shipped).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
FIRST = ["محمد", "أحمد", "خالد", "سامي", "ليلى", "رنا", "يوسف", "هالة", "عمر", "نور", "إبراهيم", "فاطمة", "عبدالله"]
LAST = ["الخطيب", "النوافلة", "العبادي", "الزعبي", "الحسن", "المصري", "الشريف", "القاسم", "الطراونة", "المجالي"]
PLACES = ["الجبيهة الشمالي", "تلاع العلي", "أبو نصير", "خلدا", "الرابية", "شفا بدران", "مرج الحمام", "ناعور"]
QUALITY = {
    "clean": {},
    "mild": {"rotate": 0.4, "blur": 0.6, "noise": 8, "jpeg": 70},
    "heavy": {"rotate": 0.9, "blur": 1.1, "noise": 18, "jpeg": 40},
}


def _line(rng: random.Random, cat: str) -> tuple[str, str]:
    """Return (full line text in logical order, the value that must be read)."""
    if cat == "ar_text":
        v = f"{rng.choice(FIRST)} {rng.choice(FIRST)} {rng.choice(LAST)}" if rng.random() < .6 else rng.choice(PLACES)
        return f"{rng.choice(['اسم المالك', 'الحوض', 'المالك', 'اسم الحوض'])}: {v}", v
    if cat == "ar_indic_number":
        n = round(rng.uniform(10, 9999), rng.choice([0, 1, 2]))
        v = str(n).replace(".", "٫").translate(AR_DIGITS)
        lab, unit = rng.choice([("المساحة الكلية", " م2"), ("المساحة", " م٢"), ("رقم القطعة", ""), ("الارتفاع", " م")])
        if lab == "رقم القطعة":
            v = str(rng.randint(1, 999)).translate(AR_DIGITS)
        return f"{lab}: {v}{unit}", v
    if cat == "ar_with_latin_value":
        v = rng.choice([f"{rng.choice(['BP', 'LIC', 'P'])}-{rng.randint(2010, 2026)}/{rng.randint(10, 9999):04d}",
                        str(rng.randint(1, 999)), f"{rng.uniform(100, 5000):.2f}"])
        lab = rng.choice(["رقم الرخصة", "رقم القطعة", "المساحة الكلية", "رقم المخطط"])
        return f"{lab}: {v}", v
    if cat == "ar_date":
        y, m, d = rng.randint(2000, 2026), rng.randint(1, 12), rng.randint(1, 28)
        v = f"{d:02d}/{m:02d}/{y}"
        if rng.random() < .5:
            v = v.translate(AR_DIGITS)
        return f"تاريخ الإصدار: {v}", v
    v = f"{rng.choice(['BP', 'LIC', 'P'])}-{rng.randint(2010, 2026)}/{rng.randint(10, 9999):04d}"
    return f"{rng.choice(['Permit No', 'Licence No', 'Plot No'])}: {v}", v


def _glyphs_ok(font_path: Path, text: str) -> bool:
    from fontTools.ttLib import TTFont
    cmap = TTFont(str(font_path)).getBestCmap()
    return all(ord(c) in cmap or c.isspace() for c in text)


def render_line(text: str, font: Path, path: Path, *, px: int = 34, rotate=0.0, blur=0.0, noise=0.0, jpeg=None, seed=0):
    import io

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    fnt = ImageFont.truetype(str(font), px, layout_engine=ImageFont.Layout.RAQM)
    rtl = sum("؀" <= c <= "ۿ" for c in text) > sum(c.isascii() and c.isalpha() for c in text)
    d = "rtl" if rtl else "ltr"
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=fnt, direction=d)
    pad = int(px * 0.5)
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=fnt, fill=0, direction=d)
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        arr = np.asarray(img, dtype=np.float32) + np.random.default_rng(seed).normal(0, noise, img.size[::-1])
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    if jpeg:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg)
        img = Image.open(io.BytesIO(buf.getvalue()))
    img.convert("RGB").save(path)


def cmd_gen(args) -> int:
    rng = random.Random(args.seed)
    fonts = sorted(Path(args.fonts).glob("*.ttf"))
    if not fonts:
        print("no .ttf fonts in --fonts")
        return 1
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    cats = ["ar_text", "ar_indic_number", "ar_with_latin_value", "latin", "ar_date"]
    for q, deg in QUALITY.items():
        for i in range(args.n):
            cat = cats[i % len(cats)]
            text, value = _line(rng, cat)
            font = rng.choice([f for f in fonts if _glyphs_ok(f, text)])
            name = f"img/{q}_{cat}_{i:03d}.png"
            render_line(text, font, out / name, px=rng.choice([28, 34, 40]), seed=rng.randint(0, 10**6), **deg)
            rows.append({"image": name, "text": text, "value": value, "category": cat, "quality": q, "font": font.name})
    (out / "gt.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} lines to {out}")
    return 0


# ------------------------------------------------------------------ engines
PREPROCESS = ("none", "upscale2", "contrast", "otsu", "pad", "sharpen", "upscale2_contrast")


def preprocess(img, mode: str):
    """Candidate line-image clean-ups (step 1 experiments)."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    g = img.convert("L")
    if mode in ("upscale2", "upscale2_contrast"):
        g = g.resize((g.width * 2, g.height * 2), Image.BICUBIC)
    if mode in ("contrast", "upscale2_contrast"):
        g = ImageOps.autocontrast(g, cutoff=1)
    if mode == "otsu":
        a = np.asarray(g)
        p = np.bincount(a.ravel(), minlength=256).astype(float)
        p /= p.sum()
        w0 = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        between = (mu[-1] * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
        g = Image.fromarray(((a > np.argmax(between)) * 255).astype("uint8"))
    if mode == "pad":
        g = ImageOps.expand(g, border=g.height // 2, fill=255)
    if mode == "sharpen":
        g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    return g.convert("RGB")


def _engine(name: str, models_dir: str | None, prep: str = "none"):
    import numpy as np
    from PIL import Image

    def load(paths):
        return [np.asarray(preprocess(Image.open(p), prep))[:, :, ::-1].copy() for p in paths]

    if name == "paddle_merge":
        from document_parser_core.core.paddle_handler import merge_line_readings
        ar, en = _engine("paddle_ar", models_dir, prep), _engine("paddle_en", models_dir, prep)

        def merged(paths):
            out = []
            for (a, sa), (e, se) in zip(ar(paths), en(paths), strict=True):
                r = merge_line_readings(a, sa, e, se)
                out.append((r.text, min(x for x in (r.score, r.latin_score) if x is not None)))
            return out
        return merged
    if name == "paddle_vl":
        from paddleocr import DocVLM
        vl = DocVLM(model_name="PaddleOCR-VL-0.9B", model_dir=f"{models_dir}/PaddleOCR-VL", device="cpu")
        return lambda paths: [(str(vl.predict({"image": a, "query": "OCR:"})[0]["result"]), None) for a in load(paths)]
    if name == "qari":
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        path = f"{models_dir}/qari-v0.3"
        model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, local_files_only=True)
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        prompt = ("Below is the image of one page of a document, as well as some raw textual content that was "
                  "previously extracted for it. Just return the plain text representation of this document as if "
                  "you were reading it naturally. Do not hallucinate.")

        def read(paths):
            out = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], images=[img], return_tensors="pt")
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                out.append((proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), None))
            return out
        return read
    if name.startswith("paddle_"):
        from paddleocr import TextRecognition
        model = {"paddle_ar": "arabic_PP-OCRv5_mobile_rec", "paddle_en": "en_PP-OCRv5_mobile_rec"}[name]
        rec = TextRecognition(model_name=model, model_dir=f"{models_dir}/paddle/{model}", device="cpu", enable_mkldnn=False)
        return lambda paths: [(str(o["rec_text"]), float(o["rec_score"])) for o in rec.predict(load(paths))]
    raise SystemExit(f"unknown engine {name!r}")


def cmd_run(args) -> int:
    out = Path(args.out)
    gt = [json.loads(x) for x in (out / "gt.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.categories:
        keep = set(args.categories.split(","))
        gt = [g for g in gt if g["category"] in keep]
    if args.limit_per_group:
        seen: dict = {}
        gt = [g for g in gt if seen.setdefault((g["category"], g["quality"]), []).append(1) or len(seen[(g["category"], g["quality"])]) <= args.limit_per_group]
    rec = _engine(args.engine, args.models_dir, args.preprocess)
    t0 = time.perf_counter()
    preds = []
    for i in range(0, len(gt), 16):
        batch = gt[i:i + 16]
        preds += rec([out / r["image"] for r in batch])
    dt = time.perf_counter() - t0
    with open(args.pred, "w", encoding="utf-8") as fh:
        label = args.engine + ("" if args.preprocess == "none" else f"+{args.preprocess}")
        fh.write(json.dumps({"engine": label, "seconds": dt, "lines": len(gt)}) + "\n")
        for r, p in zip(gt, preds, strict=True):
            text, score = p if isinstance(p, tuple) else (p, None)
            fh.write(json.dumps({"image": r["image"], "pred": text, "score": score}, ensure_ascii=False) + "\n")
    print(f"{args.engine}: {len(gt)} lines in {dt:.1f}s")
    return 0


# ------------------------------------------------------------------ scoring
def _norm(s: str) -> str:
    from document_parser_core.text_utils import normalize_text
    s = normalize_text(unicodedata.normalize("NFKC", s))
    return re.sub(r"\s+", " ", s).strip()


def _digits_ascii(s: str) -> str:
    s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٫۰۱۲۳۴۵۶۷۸۹", "0123456789.0123456789"))
    return re.sub(r"(?<=\d)\u060c(?=\d)", ".", s)  # Arabic comma used as decimal separator


def _lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cmd_score(args) -> int:
    out = Path(args.out)
    gt = {json.loads(x)["image"]: json.loads(x) for x in (out / "gt.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    for pf in args.preds:
        lines = Path(pf).read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        stats: dict[tuple, list] = {}
        for x in lines[1:]:
            p = json.loads(x)
            g = gt[p["image"]]
            pred, ref = _norm(p["pred"]), _norm(g["text"])
            value_ok = _digits_ascii(_norm(g["value"])).replace(" ", "") in _digits_ascii(pred).replace(" ", "")
            cer = _lev(ref.replace(" ", ""), pred.replace(" ", "")) / max(1, len(ref.replace(" ", "")))
            for key in (("ALL",), ("cat", g["category"]), ("q", g["quality"])):
                stats.setdefault(key, []).append((value_ok, cer))
        print(f"\n== {meta['engine']}  ({meta['seconds'] / meta['lines'] * 1000:.0f} ms/line)")
        for key in sorted(stats, key=lambda k: (len(k), k)):
            v = stats[key]
            print(f"  {'/'.join(key):28} n={len(v):3}  value_ok={sum(a for a, _ in v) / len(v):6.1%}  cer={sum(c for _, c in v) / len(v):.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("out"); g.add_argument("--fonts", required=True)
    g.add_argument("--n", type=int, default=40); g.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run"); r.add_argument("out"); r.add_argument("--engine", required=True)
    r.add_argument("--models-dir"); r.add_argument("--pred", required=True)
    r.add_argument("--preprocess", default="none", choices=PREPROCESS)
    r.add_argument("--categories", help="comma-separated subset of categories")
    r.add_argument("--limit-per-group", type=int, help="max lines per (category, quality)")
    s = sub.add_parser("score"); s.add_argument("out"); s.add_argument("preds", nargs="+")
    args = ap.parse_args(argv)
    return {"gen": cmd_gen, "run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
