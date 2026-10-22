# Benchmark Report — document_parser_core {{package_version}}

- Date: {{date}}
- Dataset: `{{dataset}}` (manually verified, `split = benchmark`, not used for prompt/rule development)
- Options: `{{options}}`
- Hardware: _fill in (CPU model, cores, RAM, GPU/VRAM, OS)_
- Models: _fill in (PaddleOCR det/rec names + versions, VLM GGUF file + quantisation + mmproj, llama-cpp-python version)_

## Summary

| Metric | Value | Notes |
|---|---|---|
| Documents | {{documents}} | |
| Failed documents | {{failed_documents}} | status = failed |
| Field exact match (all) | {{field_exact_match}} | normalised comparison |
| Field exact match (critical) | {{critical_field_exact_match}} | |
| Required-field coverage | {{field_coverage}} | share of expected values returned as `extracted` |
| **False acceptance rate** | {{false_acceptance_rate}} | wrong value returned without review flag — the most important number |
| Uncertainty detection | {{uncertainty_detection_rate}} | share of wrong/missing fields that were flagged for review |
| Unnecessary review rate | {{unnecessary_review_rate}} | correct fields that were still flagged |
| Mean CER | {{mean_cer}} | only for documents with ground-truth text |
| Mean WER | {{mean_wer}} | |
| Coordinate numeric accuracy | {{coordinate_numeric_accuracy}} | within the configured absolute tolerance |
| Seconds / document (mean) | {{seconds_mean}} | |
| Seconds / document (p95) | {{seconds_p95}} | |
| Peak RSS (process) | {{process_peak_rss_mb}} MB | whole run, includes model loading |
| Peak Python heap (max doc) | {{max_python_peak_mb}} MB | tracemalloc; excludes native allocations |

## Breakdown to fill in

- Per document class (text PDF / scanned PDF / photo / drawing / handwriting).
- Per language (Arabic / English / mixed).
- Per routing level reached (1 / 2 / 3) and time per level.

## Observed failure modes

_List concrete failures with file names, field, expected vs. got, and the routing stage._

## Caveats

- Results apply only to this dataset, these model files and this hardware.
- `confidence` values in outputs are provider scores, not calibrated probabilities.
