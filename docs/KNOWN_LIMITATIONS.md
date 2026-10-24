# Known Limitations and Unsupported Cases

## Verification status

- **PaddleOCR (verified with real weights, 2026-10-24):** runs on CPU, offline, at ~2–4 s per
  page. Measured only on **synthetic** scans (one font, generated text, simulated
  blur/noise/skew). See `docs/BENCHMARK_RESULTS.md`. **No real-world document has been
  tested**; real scans (other fonts, stamps, photos, handwriting) will be harder.
- **Vision model (Qwen2.5-VL via llama.cpp): never run with weights.** JSON-schema
  grammar conversion and argument signatures are verified, inference is not. Exact GGUF
  file names are unconfirmed.
- No GPU testing; no Windows or macOS testing.

## Known OCR weaknesses (measured on the synthetic benchmark)

- **Arabic documents need human review for most fields** in OCR: across 8 fonts only
  ~36 % of Arabic-permit fields were accepted automatically (all others flagged).
  Arabic words lose letters in some fonts (`الزعبي` → `الزعي`); no second reader exists
  for Arabic words in the default (light) configuration.

- Some wrong values are still accepted without review: Latin letters misread consistently
  by *both* recognisers (e.g. `LIC` → `LC` on heavy blur) cannot be caught by the
  cross-check. See the false-acceptance rate in BENCHMARK_RESULTS.md.
- Numbers in Arabic-Indic digits read by OCR are always sent to review: the Arabic model
  drops the decimal separator `٫` and units.
- Mixed Arabic/Latin lines lose spaces/colons between scripts; handled for label/value
  pairs, but free text from such lines may have run-order errors.
- Pages with tilted lines can merge English and Arabic lines into one visual row; each
  segment keeps its own direction, but strong skew (> ~2°) is untested.

## Deferred by design (scope control)

- Surya layout / reading order: interface and readiness check only. Level 2 records it as `unavailable`.
- PaddleOCR table-structure pipelines (PP-StructureV3, table_recognition_v2): not integrated.
- DWG/DXF: unsupported (clear error). A CAD provider can plug in at the `core/base_handler.py` level.
- Multi-frame TIFF, HEIC, Office documents: unsupported.

## Extraction limitations

- Field extraction is label-based. A value is only found if one of the field's labels
  starts a text segment or is followed by `:`. Unusual labels need custom `fields` or the VLM.
- Values continuing over several lines are cut at the first line.
- In right-to-left rows, multi-word *English* phrases are ordered right-to-left by word
  position (single tokens such as numbers and IDs are unaffected).
- OCR word boxes are interpolated from line boxes (`approx_bbox: true`).
- Dates: only numeric `Y-M-D`, `D/M/Y` (or `M/D/Y` via `date_order`). Hijri dates and
  month names are not parsed; they are reported as invalid, never converted.
- `1,250` with `decimal_separator="auto"` is read as 1250 with a warning.
- Dunum values are not converted unless `dunum_m2` is set explicitly.
- Document-type detection is a keyword count, not a classifier; its score is not a probability.
- Coordinate tables: detected from row geometry (≥2 consecutive rows with the same
  number of numeric cells). Column meaning comes only from an explicit header, and tables
  without a recognised header are flagged for review. No CRS, datum or plausibility check
  is made, and none will be inferred.
- Visual-order Arabic text layers are detected and sent to OCR, but not repaired. Without OCR the
  page is `review_required`.
- Text-quality signals (reversed Arabic, broken glyph maps, image or vector dominated pages)
  are heuristics with fixed thresholds that were tuned on synthetic examples only.

## Difficult cases (expect review, not success)

- Handwriting, stamps and signatures over text, low-resolution phone photos, skewed scans
  (document unwarping/orientation models are disabled because they are not provisioned).
- Engineering drawings: only title-block-style label/value text is targeted, and geometry is not interpreted.
- Vertical text.

## Operational

- `peak_rss_mb` is the process lifetime peak, not per document.
- The in-process network guard cannot see subprocess workers. Use the namespace script
  or a disconnected machine for system-level proof.
- The PDF backend is AGPL-licensed (see LICENSES.md).
