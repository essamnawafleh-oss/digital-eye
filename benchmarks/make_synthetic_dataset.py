"""Generate a SYNTHETIC scanned-document dataset with ground truth.

    python benchmarks/make_synthetic_dataset.py OUT_DIR [--n 12] [--seed 7]

Renders building permits (Arabic, English, mixed) and coordinate tables as scanned
images/PDFs at three quality levels (clean, mild, heavy) with Pillow + libraqm, and
writes OUT_DIR/manifest.json for benchmarks/evaluate.py.

Results on this data measure the pipeline on *generated* pages with one font. They are
not a substitute for manually verified real documents, which are usually harder.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_parser_core.tests.fixtures import render_scan  # noqa: E402

FIRST = ["محمد", "أحمد", "خالد", "سامي", "ليلى", "رنا", "يوسف", "هالة", "عمر", "نور"]
LAST = ["الخطيب", "النوافلة", "العبادي", "الزعبي", "الحسن", "المصري", "الشريف", "القاسم"]
BASINS = ["الجبيهة الشمالي", "تلاع العلي", "أبو نصير", "خلدا", "الرابية", "شفا بدران"]
EN_NAMES = ["Omar Haddad", "Lina Saleh", "Sami Nasser", "Rana Odeh", "Yousef Amin"]
QUALITY = {
    "clean": {},
    "mild": {"rotate": 0.6, "blur": 0.6, "noise": 8, "jpeg_quality": 70},
    "heavy": {"rotate": 1.5, "blur": 1.1, "noise": 18, "jpeg_quality": 40},
}
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def permit(rng: random.Random, variant: str):
    year = rng.randint(2015, 2026)
    month, day = rng.randint(1, 12), rng.randint(13, 28)  # day > 12: unambiguous D/M
    permit_no = f"{rng.choice(['BP', 'LIC', 'P'])}-{year}/{rng.randint(10, 9999):04d}"
    plot = str(rng.randint(1, 999))
    area = round(rng.uniform(250, 5000), rng.choice([0, 1, 2]))
    area_txt = f"{area:,.2f}" if area >= 1000 and rng.random() < 0.5 else f"{area}"
    owner_ar = f"{rng.choice(FIRST)} {rng.choice(FIRST)} {rng.choice(LAST)}"
    basin = rng.choice(BASINS)
    iso = f"{year}-{month:02d}-{day:02d}"
    if variant == "ar":
        area_ar = f"{area}".replace(".", "٫").translate(AR_DIGITS)
        rows = ["رخصة بناء", f"رقم الرخصة: {permit_no}", f"تاريخ الإصدار: {day:02d}/{month:02d}/{year}",
                f"اسم المالك: {owner_ar}", f"رقم القطعة: {plot}", f"الحوض: {basin}", f"المساحة الكلية: {area_ar} م2"]
        expected = {"permit_number": permit_no, "issue_date": iso, "owner_name": owner_ar, "plot_number": plot,
                    "basin": basin, "area": {"value": float(area), "unit": "m2"}}
    elif variant == "en":
        owner = rng.choice(EN_NAMES)
        rows = ["Building Permit", f"Permit No: {permit_no}", f"Issue Date: {day:02d}/{month:02d}/{year}",
                f"Owner Name: {owner}", f"Plot No: {plot}", f"Total Area: {area_txt} m2"]
        expected = {"permit_number": permit_no, "issue_date": iso, "owner_name": owner, "plot_number": plot,
                    "basin": None, "area": {"value": float(area), "unit": "m2"}}
    else:  # mixed
        rows = ["Building Permit / رخصة بناء", f"Permit No: {permit_no}", f"Issue Date: {iso}",
                f"اسم المالك: {owner_ar}", f"Plot No: {plot}", f"الحوض: {basin}", f"Total Area: {area_txt} m2"]
        expected = {"permit_number": permit_no, "issue_date": iso, "owner_name": owner_ar, "plot_number": plot,
                    "basin": basin, "area": {"value": float(area), "unit": "m2"}}
    return rows, expected


def coord_table(rng: random.Random):
    e0, n0 = rng.uniform(200000, 260000), rng.uniform(1100000, 1180000)
    rows, gt = [], []
    for i in range(1, rng.randint(4, 8) + 1):
        e, n, z = round(e0 + rng.uniform(-80, 80), 2), round(n0 + rng.uniform(-80, 80), 2), round(rng.uniform(700, 1100), 2)
        rows.append([str(i), f"{e:.2f}", f"{n:.2f}", f"{z:.2f}"])
        gt.append([str(i), e, n, z])
    return ["Coordinates of plot corners", "", ["Point", "E", "N", "Z"], *rows], gt


def _covers(font: Path, rows) -> bool:
    from fontTools.ttLib import TTFont
    cmap = TTFont(str(font)).getBestCmap()
    text = "".join(r if isinstance(r, str) else "".join(r) for r in rows)
    return all(ord(c) in cmap or c.isspace() for c in text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=12, help="permits per quality level")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fonts", help="folder of .ttf fonts to pick from per document (default: one system font)")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    fonts = sorted(Path(args.fonts).glob("*.ttf")) if args.fonts else [None]

    def pick_font(rows):
        ok = [f for f in fonts if f is None or _covers(f, rows)]
        return rng.choice(ok or [None])
    (args.out / "docs").mkdir(parents=True, exist_ok=True)
    manifest = []
    for quality, deg in QUALITY.items():
        for i in range(args.n):
            variant = ["ar", "en", "mixed"][i % 3]
            rows, expected = permit(rng, variant)
            ext = rng.choice([".png", ".jpg", ".pdf"])
            name = f"docs/permit_{quality}_{variant}_{i:02d}{ext}"
            font = pick_font(rows)
            render_scan(rows, args.out / name, font=font, seed=rng.randint(0, 10**6), **deg)
            manifest.append({"file": name, "split": "benchmark", "document_type": "building_permit", "font": getattr(font, "name", "default"),
                             "critical_fields": ["permit_number", "area", "plot_number"],
                             "expected_fields": expected, "tags": [quality, variant, "synthetic"]})
        for i in range(max(1, args.n // 4)):
            rows, gt = coord_table(rng)
            name = f"docs/coords_{quality}_{i:02d}.png"
            font = pick_font(rows)
            render_scan(rows, args.out / name, font=font, seed=rng.randint(0, 10**6), **deg)
            manifest.append({"file": name, "split": "benchmark", "document_type": "coordinate_table", "font": getattr(font, "name", "default"),
                             "expected_fields": {"coordinate_system": None},
                             "expected_tables": [{"rows": gt}], "tags": [quality, "table", "synthetic"]})
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest)} documents to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
