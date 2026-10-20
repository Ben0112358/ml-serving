import time
from types import SimpleNamespace

from ml_serving.board_meeting_protocol import backends
from ml_serving.board_meeting_protocol.backends import (
    _CursorCall,
    _read_chat_response,
    _reply_from_cursor,
)
from ml_serving.board_meeting_protocol.jobs import JobStore


class _Body:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._read = False

    def readline(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        line, _, rest = self._raw.partition(b"\n")
        self._rest = rest
        return line + b"\n"

    def read(self) -> bytes:
        return getattr(self, "_rest", b"")

    def __iter__(self):
        yield from self._rest.splitlines(keepends=True)


def test_trace_keeps_thinking_and_what_the_orchestrator_received():
    store = JobStore()
    record = store.create()
    store.note(
        record,
        {"kind": "start", "step": "row-1", "title": "History row 1 of 1"},
    )
    store.note(
        record,
        {
            "kind": "delta",
            "step": "row-1",
            "channel": "thinking",
            "text": "compare",
        },
    )
    store.note(record, {"kind": "done", "step": "row-1", "reply": "summary"})
    store.note(
        record,
        {
            "kind": "start",
            "step": "orchestrator",
            "title": "Orchestrator",
            "received": "Row 1:\nsummary",
        },
    )
    public = store.to_public(record)
    assert public["trace"][0]["thinking"] == "compare"
    assert public["trace"][0]["reply"] == "summary"
    assert public["trace"][0]["state"] == "done"
    assert public["trace"][1]["received"] == "Row 1:\nsummary"
    assert (
        "thinking" in public["message"]
        or public["trace"][1]["state"] == "running"
    )


def test_cursor_model_list_is_reused_while_a_run_is_busy(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    backends._CURSOR_MODELS.clear()
    backends._CURSOR_MODELS.extend(["composer-2.5-fast", "grok-4.7"])
    backends._CURSOR_MODELS_AT = time.monotonic()

    def fail():
        raise RuntimeError("executor busy")

    monkeypatch.setattr(
        "cursor_sdk.Cursor.models.list",
        fail,
        raising=False,
    )
    assert backends.cursor_model_ids() == ["composer-2.5-fast", "grok-4.7"]


def test_sse_splits_thinking_and_reply():
    seen: list[tuple[str, str]] = []
    raw = (
        b'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}\n'
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        b"data: [DONE]\n"
    )
    text = _read_chat_response(
        _Body(raw), lambda channel, piece: seen.append((channel, piece))
    )
    assert text == "Hello"
    assert seen == [("thinking", "hmm"), ("reply", "Hello")]


def _delta(kind: str, text: str = ""):
    return SimpleNamespace(type=kind, text=text)


def test_cursor_call_keeps_the_first_email():
    seen: list[tuple[str, str]] = []
    call = _CursorCall(lambda channel, piece: seen.append((channel, piece)))
    call.handle(_delta("thinking-delta", "think"))
    call.handle(_delta("text-delta", "I'll check.\nÄmne: One\nHej\n"))
    call.handle(_delta("text-delta", "Ämne: Two\nHej igen\n"))
    assert call.stopped
    assert call.raw().strip() == "I'll check.\nÄmne: One\nHej"
    assert _reply_from_cursor("", call.raw(), call.stopped) == "Ämne: One\nHej"
    assert ("reply", "Ämne: Two\nHej igen\n") not in seen
    assert seen[0] == ("thinking", "think")


def test_cursor_call_stops_after_the_first_turn_with_text():
    call = _CursorCall(None)
    call.handle(_delta("thinking-delta", "still working"))
    call.handle(_delta("turn-ended"))
    assert not call.stopped
    call.handle(_delta("text-delta", "Hello"))
    call.handle(_delta("turn-ended"))
    call.handle(_delta("text-delta", " again"))
    call.handle(_delta("thinking-delta", "more"))
    assert call.raw() == "Hello"
    assert _reply_from_cursor("", call.raw(), call.stopped) == "Hello"


def test_empty_cursor_result_uses_the_streamed_reply():
    assert _reply_from_cursor("", "Ämne: One\nHej", False) == "Ämne: One\nHej"
