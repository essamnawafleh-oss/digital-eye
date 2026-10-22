# Known Limitations and Unsupported Cases

## Not verified (blocked by the development environment)

- **No real model has been run.** Hugging Face, Baidu BOS, ModelScope, AIStudio and
  datalab.to were unreachable, so PaddleOCR and Qwen2.5-VL code paths are tested only
  for configuration, readiness checks, error handling and output validation (with fake
  providers). Inference behaviour, Arabic recognition quality, PaddleOCR's Arabic output
  direction, VLM JSON compliance and speed are all **unknown**.
- Exact Qwen2.5-VL GGUF file names / checksums are unconfirmed.
- No GPU testing; no Windows or macOS testing.
- **No accuracy figures exist.** Run `benchmarks/evaluate.py` on a manually verified dataset first.

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
