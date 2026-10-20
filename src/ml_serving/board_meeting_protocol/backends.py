import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request

_CURSOR_MODELS: list[str] = []
_CURSOR_MODELS_AT = 0.0
_CURSOR_MODELS_TTL = 600.0


class OpenAICompatibleModel:
    def __init__(self) -> None:
        self.base_url = (
            os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
        )
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.default_model = os.environ.get("OPENAI_MODEL", "").strip()

    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.default_model)

    def complete(self, prompt: str, model: str | None, on_delta=None) -> str:
        if not self.available():
            raise RuntimeError("OpenAI-compatible backend is not configured")
        chosen = model or self.default_model
        payload = {
            "model": chosen,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return _read_chat_response(response, on_delta)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed: {detail}") from exc


class CursorLocalModel:
    def available(self) -> bool:
        return bool(os.environ.get("CURSOR_API_KEY", "").strip())

    def complete(self, prompt: str, model: str | None, on_delta=None) -> str:
        if not self.available():
            raise RuntimeError("Cursor backend is not configured")
        if not model:
            raise RuntimeError("Choose a model for this Cursor call")
        try:
            from cursor_sdk import (
                Agent,
                AgentOptions,
                LocalAgentOptions,
                SendOptions,
            )
        except ImportError as exc:
            raise RuntimeError("Cursor backend is not installed") from exc
        call = _CursorCall(on_delta)
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent.create(
                AgentOptions(
                    api_key=os.environ["CURSOR_API_KEY"],
                    model=model,
                    tools=[],
                    local=LocalAgentOptions(cwd=directory),
                )
            )
            try:
                run = agent.send(prompt, SendOptions(on_delta=call.handle))
                call.attach(run)
                try:
                    result = run.wait()
                except Exception:
                    result = None
            finally:
                agent.close()
        official = str(getattr(result, "result", None) or "")
        return _reply_from_cursor(official, call.raw(), call.stopped)


_DRAFT_START = re.compile(r"(?m)^(Ämne|Subject)\s*:")


def _one_draft(text: str) -> str:
    matches = list(_DRAFT_START.finditer(text))
    if not matches:
        return text.strip()
    start = matches[0].start()
    end = matches[1].start() if len(matches) > 1 else len(text)
    return text[start:end].strip()


def _reply_from_cursor(result_text: str, streamed: str, stopped: bool) -> str:
    official = _one_draft(result_text)
    live = _one_draft(streamed)
    if stopped and live:
        text = live
    else:
        text = official or live
    if not text:
        raise RuntimeError("Cursor backend returned an empty reply")
    return text


class _CursorCall:
    """One local agent call. Later turns and a second draft are dropped."""

    def __init__(self, on_delta) -> None:
        self._on_delta = on_delta
        self._parts: list[str] = []
        self._run = None
        self.stopped = False
        self._cancel_sent = False

    def attach(self, run) -> None:
        self._run = run
        if self.stopped:
            self._cancel()

    def raw(self) -> str:
        return "".join(self._parts)

    def handle(self, update) -> None:
        if self.stopped:
            return
        kind = getattr(update, "type", "")
        text = getattr(update, "text", "") or ""
        if kind == "thinking-delta":
            self._emit("thinking", text)
            return
        if kind == "text-delta":
            self._take_reply(text)
            return
        if kind == "turn-ended" and self.raw().strip():
            self._halt()

    def _take_reply(self, piece: str) -> None:
        if not piece:
            return
        combined = self.raw() + piece
        matches = list(_DRAFT_START.finditer(combined))
        if len(matches) < 2:
            self._parts.append(piece)
            self._emit("reply", piece)
            return
        end = matches[1].start()
        extra = combined[:end].removeprefix(self.raw())
        if extra:
            self._parts.append(extra)
            self._emit("reply", extra)
        self._halt()

    def _emit(self, channel: str, text: str) -> None:
        if text and self._on_delta is not None:
            self._on_delta(channel, text)

    def _halt(self) -> None:
        self.stopped = True
        self._cancel()

    def _cancel(self) -> None:
        if self._cancel_sent or self._run is None:
            return
        self._cancel_sent = True
        try:
            self._run.cancel()
        except Exception:
            return


def _read_chat_response(response, on_delta) -> str:
    first = response.readline()
    if not first.lstrip().startswith(b"data:"):
        raw = first + response.read()
        return _chat_json(json.loads(raw.decode("utf-8")), on_delta)
    parts: list[str] = []
    _consume_sse_line(first, parts, on_delta)
    for line in response:
        if _consume_sse_line(line, parts, on_delta):
            break
    return "".join(parts)


def _consume_sse_line(line: bytes, parts: list[str], on_delta) -> bool:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return False
    data = text[5:].strip()
    if data == "[DONE]":
        return True
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    choices = payload.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    _emit_piece(
        delta.get("reasoning_content") or delta.get("reasoning"),
        "thinking",
        on_delta,
    )
    content = delta.get("content") or ""
    if content:
        parts.append(content)
        _emit_piece(content, "reply", on_delta)
    return False


def _chat_json(body: dict, on_delta) -> str:
    message = body["choices"][0]["message"]
    reasoning = (
        message.get("reasoning_content") or message.get("reasoning") or ""
    )
    content = message.get("content") or ""
    _emit_piece(reasoning, "thinking", on_delta)
    _emit_piece(content, "reply", on_delta)
    return content


def _emit_piece(text, channel: str, on_delta) -> None:
    if text and on_delta is not None:
        on_delta(channel, text)


def cursor_model_ids() -> list[str]:
    if not os.environ.get("CURSOR_API_KEY", "").strip():
        return []
    now = time.monotonic()
    if _CURSOR_MODELS and now - _CURSOR_MODELS_AT < _CURSOR_MODELS_TTL:
        return list(_CURSOR_MODELS)
    try:
        from cursor_sdk import Cursor
    except ImportError:
        return list(_CURSOR_MODELS)
    try:
        listed = Cursor.models.list()
    except Exception:
        return list(_CURSOR_MODELS)
    found = _ids_from(listed)
    if found:
        _remember_cursor_models(found)
    return found or list(_CURSOR_MODELS)


def _remember_cursor_models(ids: list[str]) -> None:
    global _CURSOR_MODELS_AT
    _CURSOR_MODELS.clear()
    _CURSOR_MODELS.extend(ids)
    _CURSOR_MODELS_AT = time.monotonic()


def _ids_from(listed: object) -> list[str]:
    if listed is None:
        return []
    if isinstance(listed, dict):
        for key in ("models", "data", "items"):
            if key in listed:
                return _ids_from(listed[key])
        return []
    if isinstance(listed, str):
        return [listed]
    if isinstance(listed, list):
        ids: list[str] = []
        for item in listed:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
            elif hasattr(item, "id"):
                ids.append(str(item.id))
        return ids
    if hasattr(listed, "models"):
        return _ids_from(listed.models)
    return []
