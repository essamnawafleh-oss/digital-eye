"""Check the local model layout (no network) and pin files by SHA-256.

    python scripts/verify_models.py [--models-dir DIR] [--write]

Exit code 0 = the OCR models are present (the vision model only with --require-vlm)
and match manifest.json if one exists; 1 = something is missing or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_parser_core.config import ParserConfig  # noqa: E402
from document_parser_core.core.paddle_handler import PaddleOCRProvider  # noqa: E402
from document_parser_core.core.vlm_handler import LlamaCppVLMProvider  # noqa: E402
from document_parser_core.exceptions import DocumentParserError  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir")
    ap.add_argument("--write", action="store_true", help="write manifest.json with checksums")
    ap.add_argument("--require-vlm", action="store_true", help="fail if the optional vision model is missing")
    args = ap.parse_args(argv)
    cfg = ParserConfig.from_env()
    if args.models_dir:
        cfg.models_dir = args.models_dir
    if not cfg.models_dir:
        print("Set --models-dir or DPC_MODELS_DIR")
        return 1
    root = Path(cfg.models_dir)
    ok = True
    for provider, required in ((PaddleOCRProvider(cfg), True), (LlamaCppVLMProvider(cfg), args.require_vlm)):
        try:
            provider.check_ready()
            print(f"[ok]       {provider.name}")
        except DocumentParserError as exc:
            ok = ok and not required
            print(f"[{'missing' if required else 'optional'}]{' ' if required else ''} {provider.name}: {exc}")

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts)
    current = {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in files}
    manifest = root / "manifest.json"
    if args.write:
        manifest.write_text(json.dumps(current, indent=2))
        print(f"wrote {manifest} ({len(current)} files)")
    elif manifest.exists():
        pinned = json.loads(manifest.read_text())
        for name, digest in pinned.items():
            if current.get(name) != digest:
                ok = False
                print(f"[changed] {name}: {'missing' if name not in current else 'checksum differs'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
