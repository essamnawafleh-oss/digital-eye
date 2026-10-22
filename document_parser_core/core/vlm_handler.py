"""Vision-language provider: llama-cpp-python (0.3.36) + Qwen2.5-VL GGUF.

Source-verified facts this module relies on (llama_cpp/llama_chat_format.py):
* `Qwen25VLChatHandler(clip_model_path=<mmproj.gguf>)` exists and uses libmtmd.
* Images are passed as `image_url`; anything that is not a `data:` URI is fetched with
  `urllib.request.urlopen`. We therefore ONLY send base64 `data:` URIs.
* `response_format={"type": "json_object", "schema": ...}` is converted to a grammar;
  if conversion fails it silently falls back to generic JSON, so the output is always
  re-validated here.

End-to-end behaviour with real weights is UNVERIFIED in the development environment
(model hosts were unreachable). Treat VLM answers as candidates that need
corroboration; the router never accepts an uncorroborated VLM value without review.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image

from ..config import ParserConfig, apply_offline_env
from ..exceptions import (
    MalformedModelOutputError,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
)
from .base_handler import VLMFieldAnswer, VLMProvider, VLMResponse

VLM_STATUSES = ("found", "not_found", "illegible", "explicitly_absent")

SYSTEM_PROMPT = (
    "You are a strict document data extraction tool. You read ONLY what is visibly "
    "written in the image. Rules: (1) Never guess, infer, compute or complete a value. "
    "(2) If a field is not visible, use status \"not_found\" and value null. "
    "(3) If the field is visible but you cannot read it with certainty, use status "
    "\"illegible\" and value null. (4) If the document explicitly states the value is "
    "absent (e.g. 'N/A', 'لا يوجد', '-'), use status \"explicitly_absent\". "
    "(5) evidence_text must be the exact characters copied from the image that contain "
    "the value (keep the original language and digits). (6) Answer with JSON only."
)


def build_output_schema(field_names: list[str]) -> dict[str, Any]:
    answer = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": list(VLM_STATUSES)},
            "value": {"type": ["string", "null"]},
            "evidence_text": {"type": ["string", "null"]},
        },
        "required": ["status", "value", "evidence_text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": {name: answer for name in field_names},
                "required": list(field_names),
                "additionalProperties": False,
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


def build_prompt(fields: list[dict[str, Any]], document_hint: str | None) -> str:
    lines = []
    if document_hint:
        lines.append(f"Document type (may be wrong): {document_hint}.")
    lines.append("Extract these fields from the page image:")
    for f in fields:
        desc = f.get("description") or ""
        labels = ", ".join(f.get("labels", [])[:6])
        lines.append(f"- {f['name']} ({f.get('type', 'text')}): {desc} Typical labels: {labels}")
    lines.append('Return {"fields": {<name>: {"status", "value", "evidence_text"}}} for every field.')
    return "\n".join(lines)


def parse_vlm_output(raw: str, field_names: list[str]) -> dict[str, VLMFieldAnswer]:
    """Strictly validate model output. Raises MalformedModelOutputError."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedModelOutputError(f"VLM output is not valid JSON: {exc}; output={raw[:200]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("fields"), dict):
        raise MalformedModelOutputError("VLM output must be an object with a 'fields' object")
    answers: dict[str, VLMFieldAnswer] = {}
    unknown = set(data["fields"]) - set(field_names)
    if unknown:
        raise MalformedModelOutputError(f"VLM output contains unrequested fields: {sorted(unknown)}")
    for name in field_names:
        item = data["fields"].get(name)
        if not isinstance(item, dict):
            raise MalformedModelOutputError(f"VLM output missing field '{name}'")
        status = item.get("status")
        value = item.get("value")
        evidence = item.get("evidence_text")
        if status not in VLM_STATUSES:
            raise MalformedModelOutputError(f"Field '{name}': invalid status {status!r}")
        if value is not None and not isinstance(value, (str, int, float)):
            raise MalformedModelOutputError(f"Field '{name}': value must be string or null")
        if evidence is not None and not isinstance(evidence, str):
            raise MalformedModelOutputError(f"Field '{name}': evidence_text must be string or null")
        if status == "found" and (value is None or str(value).strip() == ""):
            raise MalformedModelOutputError(f"Field '{name}': status 'found' requires a value")
        if status != "found":
            value = None  # never keep a value the model itself did not claim to see
        answers[name] = VLMFieldAnswer(
            name=name,
            status=status,
            value=None if value is None else str(value),
            evidence_text=evidence,
        )
    return answers


def image_to_data_uri(image: Image.Image, max_side: int) -> str:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class LlamaCppVLMProvider(VLMProvider):
    name = "llama_cpp_vlm"

    def __init__(self, config: ParserConfig):
        self.config = config
        self._llm = None

    def check_ready(self) -> None:
        cfg = self.config
        if cfg.vlm.chat_handler != "qwen2.5-vl":
            raise ProviderUnavailableError(
                "vlm", f"chat_handler '{cfg.vlm.chat_handler}' is not verified",
                "only 'qwen2.5-vl' is implemented",
            )
        for path, what in ((cfg.vlm_model_path(), "GGUF language model"), (cfg.vlm_mmproj_path(), "multimodal projector (mmproj) GGUF")):
            if path is None:
                raise ModelNotFoundError("vlm", "<unset>", what, "Set models_dir or vlm.model_path / vlm.mmproj_path.")
            if not path.is_file():
                raise ModelNotFoundError("vlm", str(path), what)
        try:
            import llama_cpp  # noqa: F401
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # noqa: F401
        except ImportError as exc:
            raise ProviderUnavailableError(
                "vlm", str(exc), "pip install -r document_parser_core/requirements-vlm.txt"
            ) from exc

    def _load(self):
        if self._llm is not None:
            return self._llm
        self.check_ready()
        if self.config.offline:
            apply_offline_env()
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        v = self.config.vlm
        try:
            handler = Qwen25VLChatHandler(clip_model_path=str(self.config.vlm_mmproj_path()), verbose=False)
            self._llm = Llama(
                model_path=str(self.config.vlm_model_path()),
                chat_handler=handler,
                n_ctx=v.n_ctx,
                n_threads=v.n_threads,
                n_gpu_layers=v.n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise ProviderError(f"Failed to load VLM: {exc}") from exc
        return self._llm

    def _complete(self, messages: list[dict], schema: dict) -> str:
        llm = self._load()
        out = llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=self.config.vlm.temperature,
            max_tokens=self.config.vlm.max_tokens,
        )
        return out["choices"][0]["message"]["content"] or ""

    def extract_fields(self, image, page_number, fields, document_hint=None) -> VLMResponse:
        names = [f["name"] for f in fields]
        schema = build_output_schema(names)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(image, self.config.vlm.max_image_side_px)}},
                    {"type": "text", "text": build_prompt(fields, document_hint)},
                ],
            },
        ]
        last_error: Exception | None = None
        raw = ""
        attempts = 0
        for attempts in range(1, self.config.vlm.max_retries + 2):
            raw = self._complete(messages, schema)
            try:
                answers = parse_vlm_output(raw, names)
                return VLMResponse(answers=answers, raw_output=raw, attempts=attempts, model=self.config.vlm.model_file)
            except MalformedModelOutputError as exc:
                last_error = exc
        raise MalformedModelOutputError(f"VLM output invalid after {attempts} attempt(s): {last_error}")
