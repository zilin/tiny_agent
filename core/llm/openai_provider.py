import json
import logging
from typing import AsyncGenerator, Dict, Any, List

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    LLM Provider for OpenAI.
    Wraps AsyncOpenAI and stream accumulation logic originally found in AgentLoop.
    """

    def __init__(self, model: str, api_key: str, base_url: str = None):
        self.model = model
        api_kwargs = {"api_key": api_key}
        if base_url:
            api_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**api_kwargs)
        logger.info(f"Initialized OpenAIProvider for model: {self.model}")

    async def stream_chat(
        self, messages: List[Dict[str, Any]], tools_def: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams chat responses compatible with AgentLoop.
        Yields normalized events: `text_delta`, `tool_calls`, `token_usage`.
        """
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools_def:
            api_kwargs["tools"] = tools_def

        try:
            response_stream = await self.client.chat.completions.create(**api_kwargs)
        except Exception as e:
            yield {"type": "error", "content": f"OpenAI API Error: {str(e)}"}
            return

        tool_call_buffer = {}

        async for chunk in response_stream:
            if hasattr(chunk, "usage") and chunk.usage:
                yield {
                    "type": "token_usage",
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Text delta
            if delta.content:
                yield {"type": "text_delta", "content": delta.content}

            # Tool calls accumulation (OpenAI sends tokens piece by piece)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_buffer:
                        tool_call_buffer[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {
                                "name": (
                                    tc.function.name
                                    if tc.function and tc.function.name
                                    else ""
                                ),
                                "arguments": (
                                    tc.function.arguments
                                    if tc.function and tc.function.arguments
                                    else ""
                                ),
                            },
                        }
                    else:
                        if tc.id:
                            tool_call_buffer[idx]["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_call_buffer[idx]["function"][
                                    "name"
                                ] += tc.function.name
                            if tc.function.arguments:
                                tool_call_buffer[idx]["function"][
                                    "arguments"
                                ] += tc.function.arguments

        # Once the stream finishes, yield the fully reconstructed tool calls
        if tool_call_buffer:
            final_tool_calls = [v for k, v in sorted(tool_call_buffer.items())]
            yield {"type": "tool_calls", "tool_calls": final_tool_calls}
