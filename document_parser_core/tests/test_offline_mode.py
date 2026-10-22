"""Offline guarantees: no network, no implicit downloads, clear missing-model errors."""

import os
import shutil
import socket
import subprocess
import sys
import textwrap

import pytest

from document_parser_core import parse_document
from document_parser_core.config import OFFLINE_ENV, ParserConfig
from document_parser_core.core.paddle_handler import PaddleOCRProvider
from document_parser_core.core.vlm_handler import LlamaCppVLMProvider
from document_parser_core.exceptions import ModelNotFoundError, OfflineViolationError
from document_parser_core.offline import network_guard

MODELS_DIR = os.environ.get("DPC_MODELS_DIR")


# ---------------------------------------------------------------- guard
def test_guard_blocks_remote_connections_and_dns():
    with network_guard():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OfflineViolationError):
            s.connect(("93.184.216.34", 443))
        s.close()
        with pytest.raises(OfflineViolationError):
            socket.getaddrinfo("huggingface.co", 443)
    # restored afterwards
    assert socket.socket.connect.__name__ == "connect" and "guard" not in repr(socket.getaddrinfo)


def test_guard_allows_loopback():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    with network_guard():
        c = socket.socket()
        c.connect(("127.0.0.1", port))
        c.close()
    server.close()


def test_parse_under_strict_guard(permit_en_pdf):
    r = parse_document(str(permit_en_pdf), "building_permit",
                       {"strict_network_guard": True, "providers": {"ocr": None, "vlm": None}})
    assert r["fields"]["permit_number"]["value"] == "BP-2024/0153"


def test_offline_env_flags_applied(permit_en_pdf):
    parse_document(str(permit_en_pdf), None, {"providers": {"ocr": None, "vlm": None}})
    for k, v in OFFLINE_ENV.items():
        assert os.environ.get(k) == v


# ---------------------------------------------------------------- missing models
def test_missing_paddle_models_fail_clearly_without_download(tmp_path, scanned_permit_pdf):
    r = parse_document(str(scanned_permit_pdf), "building_permit",
                       {"models_dir": str(tmp_path), "vlm": {"enabled": False}, "strict_network_guard": True})
    check = [s for s in r["processing"]["stages"] if s["stage"] == "ocr_check"][0]
    assert check["outcome"] == "unavailable"
    assert "not found at" in check["detail"] and "never downloaded" in check["detail"]
    assert r["status"] == "review_required"


def test_paddle_model_dir_missing_files(tmp_path):
    det = tmp_path / "paddle" / "PP-OCRv5_server_det"
    det.mkdir(parents=True)
    (det / "inference.yml").write_text("x")
    cfg = ParserConfig(models_dir=str(tmp_path))
    with pytest.raises(ModelNotFoundError, match="inference.pdiparams"):
        PaddleOCRProvider(cfg).check_ready()


def test_paddle_unset_models_dir():
    with pytest.raises(ModelNotFoundError, match="DPC_MODELS_DIR"):
        PaddleOCRProvider(ParserConfig(models_dir=None)).check_ready()


def test_missing_vlm_weights(tmp_path):
    (tmp_path / "vlm").mkdir()
    (tmp_path / "vlm" / ParserConfig().vlm.model_file).write_bytes(b"GGUF")
    with pytest.raises(ModelNotFoundError, match="mmproj"):
        LlamaCppVLMProvider(ParserConfig(models_dir=str(tmp_path))).check_ready()


def test_worker_process_propagates_missing_model_error(tmp_path, scanned_permit_pdf):
    r = parse_document(str(scanned_permit_pdf), "building_permit", {
        "models_dir": str(tmp_path),
        "paddle": {"python_executable": sys.executable, "worker_timeout_s": 60},
        "vlm": {"enabled": False},
    })
    check = [s for s in r["processing"]["stages"] if s["stage"] == "ocr_check"][0]
    assert check["outcome"] == "unavailable"
    assert "[worker]" in check["detail"] and "not found" in check["detail"]


# ---------------------------------------------------------------- system-level
def _can_unshare() -> bool:
    if not shutil.which("unshare"):
        return False
    return subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _can_unshare(), reason="needs Linux user+network namespaces (unshare -rn)")
def test_parse_in_network_namespace_without_network(permit_en_pdf):
    """Runs the parser in a process that has no network interfaces at all."""
    code = textwrap.dedent(f"""
        import json, socket
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=3); print("NETWORK_AVAILABLE")
        except OSError:
            pass
        from document_parser_core import parse_document
        r = parse_document({str(permit_en_pdf)!r}, "building_permit", {{"vlm": {{"enabled": False}}, "ocr_provider": None}})
        print(json.dumps({{"status": r["status"], "pn": r["fields"]["permit_number"]["value"]}}))
    """)
    out = subprocess.run(["unshare", "-rn", sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "NETWORK_AVAILABLE" not in out.stdout
    assert '"pn": "BP-2024/0153"' in out.stdout


# ---------------------------------------------------------------- real models (opt-in)
@pytest.mark.requires_models
@pytest.mark.skipif(not MODELS_DIR, reason="DPC_MODELS_DIR not set (no local model weights)")
def test_real_paddle_ocr_offline(scanned_permit_pdf):
    pytest.importorskip("paddleocr")
    cfg = ParserConfig(models_dir=MODELS_DIR)
    try:
        PaddleOCRProvider(cfg).check_ready()
    except ModelNotFoundError as exc:
        pytest.skip(str(exc))
    r = parse_document(str(scanned_permit_pdf), "building_permit",
                       {"models_dir": MODELS_DIR, "strict_network_guard": True, "vlm": {"enabled": False}})
    assert r["pages"][0]["text_source"].startswith("ocr:")
    assert "Permit" in r["pages"][0]["text"]


@pytest.mark.requires_models
@pytest.mark.skipif(not MODELS_DIR, reason="DPC_MODELS_DIR not set (no local model weights)")
def test_real_vlm_offline(tmp_path):
    pytest.importorskip("llama_cpp")
    from document_parser_core.tests.fixtures import make_text_pdf
    cfg = ParserConfig(models_dir=MODELS_DIR)
    try:
        LlamaCppVLMProvider(cfg).check_ready()
    except ModelNotFoundError as exc:
        pytest.skip(str(exc))
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Ref. of licence 2024/0153"]])
    r = parse_document(str(pdf), "building_permit",
                       {"models_dir": MODELS_DIR, "strict_network_guard": True, "ocr_provider": None})
    vlm_stage = [s for s in r["processing"]["stages"] if s["stage"] == "vlm_extract"]
    assert vlm_stage and vlm_stage[0]["outcome"] in ("ok", "failed")
    f = r["fields"]["permit_number"]
    assert f["status"] in ("extracted", "missing", "unreadable")
    if f["status"] == "extracted":
        assert f["value"] == "2024/0153" or f["requires_review"]
