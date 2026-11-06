import json
from collections.abc import Iterator

import httpx

from ragapp.llm.base import (
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    AssistantTurn,
    Message,
    ToolCall,
    ToolSpec,
    build_user_content,
)
from ragapp.llm.tracing import traceable

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """Anthropic's Messages API (api.anthropic.com/v1/messages) — different auth
    header, different request/response shape from the OpenAI-style providers, so
    this isn't a thin variant of OpenAICompatProvider.

    Keeps one persistent httpx.Client per instance — see Embedder for why.
    """

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            timeout=120,
        )

    @traceable(run_type="llm", name="anthropic.chat_stream")
    def chat_stream(
        self,
        user_message: str,
        context: str,
        history: list[Message] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        messages = [
            *(history or []),
            {"role": "user", "content": build_user_content(user_message, context)},
        ]
        with self._client.stream(
            "POST",
            "/messages",
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt or SYSTEM_PROMPT,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[len("data: ") :])
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {}).get("text")
                    if delta:
                        yield delta

    @traceable(run_type="llm", name="anthropic.complete_with_tools")
    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system_prompt: str
    ) -> AssistantTurn:
        anthropic_messages = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                content: list[dict] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                content.extend(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in m["tool_calls"]
                )
                anthropic_messages.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                # Anthropic has no "tool" role — a tool's result goes back as a
                # tool_result content block inside a user turn.
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                        ],
                    }
                )
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        response = self._client.post(
            "/messages",
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": anthropic_messages,
                "tools": [
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in tools
                ],
            },
        )
        response.raise_for_status()
        blocks = response.json().get("content", [])
        tool_calls = [
            ToolCall(id=b["id"], name=b["name"], arguments=b["input"]) for b in blocks if b["type"] == "tool_use"
        ]
        text = "".join(b["text"] for b in blocks if b["type"] == "text") or None
        return AssistantTurn(tool_calls=tool_calls, text=text)
