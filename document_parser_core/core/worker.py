"""Out-of-process provider adapter.

Lets a provider run under a different Python interpreter (e.g. a dedicated venv for
PaddleOCR, Surya or llama-cpp) while `parse_document` stays the single API.

Protocol: newline-delimited JSON over the worker's stdin/stdout. The worker keeps the
model loaded between requests. Library output written to stdout (including from C
code) is redirected to stderr so it cannot corrupt the protocol stream.

The worker interpreter must have this package importable (e.g. `pip install -e .
--no-deps` in that venv) plus the provider's own requirements file.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
from typing import Any

from PIL import Image

from .. import exceptions as exc_mod
from ..config import ParserConfig, _merge
from ..exceptions import DocumentParserError, ProviderError
from ..schemas import Line
from .base_handler import OCRProvider, VLMFieldAnswer, VLMProvider, VLMResponse


def config_from_snapshot(snapshot: dict) -> ParserConfig:
    cfg = ParserConfig()
    _merge(cfg, snapshot, "snapshot")
    return cfg


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _rebuild_error(payload: dict) -> DocumentParserError:
    cls = next(
        (c for c in vars(exc_mod).values()
         if isinstance(c, type) and issubclass(c, DocumentParserError) and c.__name__ == payload.get("type")),
        ProviderError,
    )
    err = cls.__new__(cls)
    Exception.__init__(err, f"[worker] {payload.get('message')}")
    return err


class _WorkerClient:
    def __init__(self, kind: str, python_executable: str, config: ParserConfig, timeout_s: float):
        self.kind = kind
        self.timeout_s = timeout_s
        env = dict(os.environ)
        if config.offline:
            from ..config import OFFLINE_ENV
            env.update(OFFLINE_ENV)
        try:
            self.proc = subprocess.Popen(
                [python_executable, "-m", "document_parser_core.core.worker", kind],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                text=True, encoding="utf-8", env=env,
            )
        except OSError as exc:
            raise exc_mod.ProviderUnavailableError(kind, f"cannot start worker '{python_executable}': {exc}") from exc
        self._lock = threading.Lock()
        self.call({"op": "init", "config": config.snapshot()})

    def call(self, request: dict) -> Any:
        with self._lock:
            if self.proc.poll() is not None:
                raise ProviderError(f"{self.kind} worker exited with code {self.proc.returncode}")
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            result: list[str] = []
            reader = threading.Thread(target=lambda: result.append(self.proc.stdout.readline()), daemon=True)
            reader.start()
            reader.join(self.timeout_s)
            if reader.is_alive():
                self.proc.kill()
                raise ProviderError(f"{self.kind} worker timed out after {self.timeout_s}s")
            if not result or not result[0]:
                raise ProviderError(f"{self.kind} worker closed the connection (exit={self.proc.poll()})")
            response = json.loads(result[0])
        if not response.get("ok"):
            raise _rebuild_error(response.get("error", {}))
        return response.get("result")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call({"op": "shutdown"})
            except DocumentParserError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class SubprocessOCRProvider(OCRProvider):
    def __init__(self, config: ParserConfig, python_executable: str, name: str = "paddleocr"):
        self.name = name
        self._client = _WorkerClient("ocr", python_executable, config, config.paddle.worker_timeout_s)

    def check_ready(self) -> None:
        self._client.call({"op": "check"})

    def recognize(self, image, page_number, scale):
        data = self._client.call({"op": "recognize", "image": _encode_image(image), "page_number": page_number, "scale": scale})
        return [Line.model_validate(item) for item in data]

    def close(self) -> None:
        self._client.close()


class SubprocessVLMProvider(VLMProvider):
    name = "llama_cpp_vlm"

    def __init__(self, config: ParserConfig, python_executable: str):
        self._client = _WorkerClient("vlm", python_executable, config, config.vlm.worker_timeout_s)

    def check_ready(self) -> None:
        self._client.call({"op": "check"})

    def extract_fields(self, image, page_number, fields, document_hint=None):
        data = self._client.call({
            "op": "extract_fields", "image": _encode_image(image), "page_number": page_number,
            "fields": fields, "document_hint": document_hint,
        })
        answers = {k: VLMFieldAnswer(**v) for k, v in data["answers"].items()}
        return VLMResponse(answers=answers, raw_output=data["raw_output"], attempts=data["attempts"], model=data["model"])

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------- worker side
def _serve(kind: str) -> int:
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # anything libraries print goes to stderr
    sys.stdout = sys.stderr
    provider = None
    for raw in sys.stdin:
        if not raw.strip():
            continue
        req = json.loads(raw)
        op = req.get("op")
        try:
            if op == "init":
                cfg = config_from_snapshot(req["config"])
                if kind == "ocr":
                    from .paddle_handler import PaddleOCRProvider
                    provider = PaddleOCRProvider(cfg)
                elif kind == "vlm":
                    from .vlm_handler import LlamaCppVLMProvider
                    provider = LlamaCppVLMProvider(cfg)
                else:
                    raise ProviderError(f"unknown worker kind {kind!r}")
                result: Any = None
            elif op == "check":
                provider.check_ready()
                result = None
            elif op == "recognize":
                lines = provider.recognize(_decode_image(req["image"]), req["page_number"], req["scale"])
                result = [ln.model_dump(mode="json") for ln in lines]
            elif op == "extract_fields":
                resp = provider.extract_fields(_decode_image(req["image"]), req["page_number"], req["fields"], req.get("document_hint"))
                result = {
                    "answers": {k: vars(v) for k, v in resp.answers.items()},
                    "raw_output": resp.raw_output, "attempts": resp.attempts, "model": resp.model,
                }
            elif op == "shutdown":
                proto.write(json.dumps({"ok": True}) + "\n")
                proto.flush()
                return 0
            else:
                raise ProviderError(f"unknown op {op!r}")
            proto.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False) + "\n")
        except Exception as exc:  # report every failure to the parent, keep serving
            proto.write(json.dumps({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}) + "\n")
        proto.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_serve(sys.argv[1] if len(sys.argv) > 1 else "ocr"))
