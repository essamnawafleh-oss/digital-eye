"""Parse one file and print the JSON result.

    python -m document_parser_core.examples.parse_file FILE [--type building_permit]
           [--models-dir DIR] [--max-level 1|2|3] [--no-text] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys

from document_parser_core import parse_document


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--type", dest="document_type")
    ap.add_argument("--models-dir")
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-text", action="store_true", help="omit full page text from output")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    options = {"render_dpi": args.dpi, "routing": {"max_level": args.max_level},
               "include_page_text": not args.no_text, "strict_network_guard": True}
    if args.models_dir:
        options["models_dir"] = args.models_dir
    result = parse_document(args.file, args.document_type, options)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    print(f"status={result['status']} review_required={result['review_required']}", file=sys.stderr)
    for reason in result["review_reasons"]:
        print(f"  review: {reason}", file=sys.stderr)
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
