"""Regression test for Gemini-style one-shot tool_call streaming.

Providers like Gemini send the entire tool_call `arguments` in a single streaming
chunk. Before the fix, `AnthropicStreamWrapper` swallowed that chunk on the
start-new-block branch, leaving downstream clients with `input: {}` on the
tool_use content block.
"""

from types import SimpleNamespace
from typing import List

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)


def _oneshot_tool_call_chunk(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_test_id",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )


def _finish_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                delta=SimpleNamespace(content=None, tool_calls=None),
            )
        ],
        usage=None,
    )


def _collect(wrapper: AnthropicStreamWrapper) -> List[dict]:
    events: List[dict] = []
    while True:
        try:
            events.append(wrapper.__next__())
        except StopIteration:
            break
    return events


def test_gemini_oneshot_tool_call_emits_input_json_delta():
    """Gemini streams full args in the trigger chunk — the wrapper must
    forward them as input_json_delta so the client receives a non-empty input."""
    stream = iter(
        [
            _oneshot_tool_call_chunk(
                name="WebSearch", arguments='{"query": "Anthropic news"}'
            ),
            _finish_chunk(),
        ]
    )
    wrapper = AnthropicStreamWrapper(
        completion_stream=stream, model="gemini-3-flash-preview"
    )

    events = _collect(wrapper)

    input_deltas = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert (
        len(input_deltas) == 1
    ), f"expected 1 input_json_delta, got {len(input_deltas)}: {events}"
    assert input_deltas[0]["delta"]["partial_json"] == '{"query": "Anthropic news"}'

    tool_use_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) == 1
    # Start event still carries empty input — deltas populate it.
    assert tool_use_starts[0]["content_block"].get("input") == {}


def test_text_content_unaffected():
    """Sanity check: plain text stream still works."""
    text_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content="hello", tool_calls=None),
            )
        ],
        usage=None,
    )
    stream = iter([text_chunk, _finish_chunk()])
    wrapper = AnthropicStreamWrapper(completion_stream=stream, model="gpt-4o")
    events = _collect(wrapper)
    # No input_json_delta should be present for plain text.
    assert not any(
        e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
        for e in events
    )
