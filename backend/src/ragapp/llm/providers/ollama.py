import json
from collections.abc import Iterator

import httpx

from ragapp.config import settings
from ragapp.llm.base import (
    SYSTEM_PROMPT,
    AssistantTurn,
    Message,
    ToolCall,
    ToolSpec,
    build_user_content,
)

from ragapp.llm.tracing import traceable


class OllamaProvider:
    """Ollama's native /api/chat endpoint (not the OpenAI-compat /v1 one — that
    endpoint silently ignores `keep_alive`, which defeats the point of setting
    it).

    Keeps one persistent httpx.Client per instance — see Embedder for why.
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or settings.chat_model
        self.host = host or settings.ollama_host
        self._client = httpx.Client(base_url=self.host, timeout=120)

    @traceable(run_type="llm", name="ollama.chat_stream")
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
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "keep_alive": settings.ollama_keep_alive,
                "options": {"num_ctx": settings.ollama_chat_num_ctx},
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content")
                if delta:
                    yield delta
                if data.get("done"):
                    break

    @traceable(run_type="llm", name="ollama.complete_with_tools")
    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec], system_prompt: str
    ) -> AssistantTurn:
        ollama_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                ollama_messages.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or "",
                        "tool_calls": [
                            {"function": {"name": tc.name, "arguments": tc.arguments}}
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            elif m["role"] == "tool":
                ollama_messages.append({"role": "tool", "content": m["content"]})
            else:
                ollama_messages.append({"role": m["role"], "content": m["content"]})

        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "keep_alive": settings.ollama_keep_alive,
                "options": {"num_ctx": settings.ollama_chat_num_ctx},
                "tools": [_tool_to_ollama(t) for t in tools],
            },
        )
        response.raise_for_status()
        message = response.json()["message"]
        # Ollama's tool_calls arguments already arrive as a parsed object, not
        # a JSON string (unlike OpenAI's wire format) — no json.loads needed.
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        return AssistantTurn(tool_calls=tool_calls, text=message.get("content") or None)


def _tool_to_ollama(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }
