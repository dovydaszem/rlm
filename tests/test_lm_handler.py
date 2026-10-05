"""Tests for LMHandler using MockLM (no real LM required)."""

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.lm_handler import LMHandler
from tests.mock_lm import MockLM


def test_lm_handler_single_request():
    """Single prompt request returns success and echo-style content."""
    mock = MockLM(responses=["hello back"])
    with LMHandler(client=mock) as handler:
        request = LMRequest(prompt="hello")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert response.chat_completion is not None
    assert response.chat_completion.response == "hello back"


def test_lm_handler_batched_request():
    """Batched prompts return one response per prompt in order."""
    responses = [f"r{i}" for i in range(5)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = [f"prompt-{i}" for i in range(5)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == 5
    for i, resp in enumerate(result):
        assert resp.success, resp.error
        assert resp.chat_completion is not None
        assert resp.chat_completion.response == f"r{i}"


def test_lm_handler_batched_partial_failure():
    """One failing call returns an error for that slot; the rest still succeed."""

    def response_fn(prompt):
        if prompt == "prompt-1":
            raise RuntimeError("boom")
        return f"ok {prompt}"

    mock = MockLM(response_fn=response_fn)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = ["prompt-0", "prompt-1", "prompt-2"]
        result = send_lm_request_batched(handler.address, prompts)

    assert len(result) == 3
    assert result[0].success
    assert result[0].chat_completion.response == "ok prompt-0"
    assert not result[1].success
    assert "boom" in result[1].error
    assert result[2].success
    assert result[2].chat_completion.response == "ok prompt-2"


def test_lm_handler_batched_many_prompts_semaphore_cap():
    """Many prompts complete successfully with semaphore limiting concurrency."""
    # 50 prompts, max 4 concurrent: should still all complete
    count = 50
    responses = [f"resp-{i}" for i in range(count)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = [f"p-{i}" for i in range(count)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == count
    for i, resp in enumerate(result):
        assert resp.success, (i, resp.error)
        assert resp.chat_completion.response == f"resp-{i}"


class _LoopBoundMockLM(MockLM):
    """Like an async HTTP client whose connection pool is tied to the first event loop it ran on,
    with per-call usage equal to the prompt length."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loop = None
        self._last_tokens = 0

    async def acompletion(self, prompt):
        import asyncio

        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError("Event loop is closed")
        await asyncio.sleep(0.01 * len(prompt) % 0.05)  # finish out of order
        self._last_tokens = len(prompt)
        return f"ok {prompt}"

    def get_last_usage(self):
        from rlm.core.types import ModelUsageSummary

        return ModelUsageSummary(
            total_calls=1, total_input_tokens=self._last_tokens, total_output_tokens=0
        )


def test_lm_handler_consecutive_batches_share_event_loop():
    """A second batch must not fail on a client bound to the first batch's loop."""
    mock = _LoopBoundMockLM()
    with LMHandler(client=mock) as handler:
        for _ in range(2):
            result = send_lm_request_batched(handler.address, ["a", "b", "c"])
            assert all(r.success for r in result), [r.error for r in result]


def test_lm_handler_batched_usage_is_per_prompt():
    """Each batched completion carries its own call's usage, not the last call's."""
    mock = _LoopBoundMockLM()
    prompts = ["x" * n for n in (5, 1, 3, 2, 4)]
    with LMHandler(client=mock, batch_max_concurrent=5) as handler:
        result = send_lm_request_batched(handler.address, prompts)
    for prompt, resp in zip(prompts, result, strict=True):
        usage = resp.chat_completion.usage_summary.model_usage_summaries["mock-model"]
        assert usage.total_input_tokens == len(prompt)
