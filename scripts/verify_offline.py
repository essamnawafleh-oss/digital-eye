"""Offline smoke test (cross-platform).

    python scripts/verify_offline.py [FILE ...] [--models-dir DIR]

1. Builds a synthetic text PDF (unless files are given) and parses it with the strict
   in-process network guard enabled: any non-loopback connection or DNS lookup raises.
2. If a models dir is configured, the scanned version is parsed too, so OCR/VLM model
   loading is exercised under the guard.

The guard cannot see child processes (e.g. subprocess workers). For a system-level
proof, also run this script with networking disabled: on Linux
`scripts/verify_offline.sh`; on Windows disable the network adapter (or unplug) and run
it again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_parser_core import parse_document  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--models-dir", default=os.environ.get("DPC_MODELS_DIR"))
    args = ap.parse_args(argv)
    files = list(args.files)
    if not files:
        from document_parser_core.tests.fixtures import make_scanned_pdf, make_text_pdf
        tmp = Path(tempfile.mkdtemp())
        text_pdf = make_text_pdf(tmp / "offline_text.pdf", [["Building Permit", "Permit No: 12/2024", "Plot No: 7"]])
        files = [str(text_pdf)]
        if args.models_dir:
            files.append(str(make_scanned_pdf(text_pdf, tmp / "offline_scan.pdf")))
    options = {"strict_network_guard": True}
    if args.models_dir:
        options["models_dir"] = args.models_dir
    failures = 0
    not_run: set[str] = set()
    for f in files:
        r = parse_document(f, None, options)
        err = r.get("error") or {}
        violated = err.get("type") == "OfflineViolationError" or any(
            "Network access attempted" in (s.get("detail") or "") for s in r["processing"]["stages"])
        print(json.dumps({"file": f, "status": r["status"], "network_violation": violated,
                          "providers": r["processing"]["providers_used"],
                          "stages": [(s["stage"], s["outcome"]) for s in r["processing"]["stages"]]}, ensure_ascii=False))
        failures += violated or r["status"] == "failed"
        not_run.update(s["stage"] for s in r["processing"]["stages"]
                       if s["stage"] in ("ocr_check", "vlm_check") and s["outcome"] == "unavailable")
    if failures:
        print("OFFLINE CHECK: FAIL")
        return 1
    if args.models_dir and not_run:
        print(f"OFFLINE CHECK: INCOMPLETE - no network use detected, but {sorted(not_run)} could not run "
              "(models missing?), so model loading was NOT verified offline")
        return 3
    print("OFFLINE CHECK: PASS" + ("" if args.models_dir else " (native text path only; pass --models-dir to cover OCR/VLM)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
