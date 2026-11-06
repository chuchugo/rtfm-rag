import json
from collections.abc import Iterator

import httpx

from ragapp.llm.base import (
    SYSTEM_PROMPT,
    AssistantTurn,
    Message,
    ToolCall,
    ToolSpec,
    build_user_content,
)

from ragapp.llm.tracing import traceable


class OpenAICompatProvider:
    """Works against OpenAI itself, or any endpoint that mirrors OpenAI's
    /chat/completions API — many cloud and self-hosted providers do (vLLM,
    LM Studio, OpenRouter, etc.). Point base_url elsewhere to use those.

    Keeps one persistent httpx.Client per instance — see Embedder for why.
    """

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )

    @traceable(run_type="llm", name="openai.chat_stream")
    def chat_stream(
        self,
        user_message: str,
        context: str,
        history: list[Message] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": build_user_content(user_message, context)},
        ]
        with self._client.stream(
            "POST",
            "/chat/completions",
            json={"model": self.model, "messages": messages, "stream": True},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :].strip()
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    @traceable(run_type="llm", name="openai.complete_with_tools")
    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system_prompt: str
    ) -> AssistantTurn:
        oai_messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                oai_messages.append(
                    {
                        "role": "assistant",
                        "content": m.get("content"),
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            elif m["role"] == "tool":
                oai_messages.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                oai_messages.append({"role": m["role"], "content": m["content"]})

        response = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": oai_messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
                    }
                    for t in tools
                ],
            },
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        # Unlike Ollama, OpenAI's tool_calls arguments arrive as a JSON string,
        # not an already-parsed object — needs explicit decoding.
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"]))
            for tc in (message.get("tool_calls") or [])
        ]
        return AssistantTurn(tool_calls=tool_calls, text=message.get("content"))
