import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import os

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    LLM Provider for Google Gemini using the `google-genai` SDK.
    Provides an interface compatible with `AgentLoop`.
    """

    def __init__(
        self,
        model: str,
        api_key: str = None,
        vertexai: bool = False,
        project: str = None,
        location: str = None,
    ):
        self.model = model

        if vertexai:
            kwargs = {}
            if project:
                kwargs["project"] = project
            if location:
                kwargs["location"] = location
            self.client = genai.Client(
                vertexai=True, http_options={"api_version": "v1beta1"}, **kwargs
            )
            logger.info(
                f"Initialized GeminiProvider (Vertex AI) for model: {self.model}"
            )
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY is required for Gemini provider (when vertexai=False)"
                )
            self.client = genai.Client(api_key=key)
            logger.info(f"Initialized GeminiProvider (API Key) for model: {self.model}")

    def _convert_tool_to_gemini(self, openai_tool: Dict[str, Any]) -> types.Tool:
        """Convert OpenAI function format to Gemini Tool format"""
        func = openai_tool.get("function", {})

        # We need to map the JSON schema to Gemini's expected format.
        # google-genai accepts standard JSON schema dicts for parameters
        parameters: Optional[Dict[str, Any]] = func.get("parameters")

        gemini_func = types.FunctionDeclaration(
            name=func.get("name", ""),
            description=func.get("description", ""),
            # Passing the raw type/properties dict often works directly in the new SDK
            # But let's be explicitly careful if parameters is empty or None
            # The SDK handles parameter dicts under the hood.
        )

        # In `google-genai`, you assign a dictionary to parameters or use the dedicated schema objects.
        # Handing the raw json schema dict to `parameters` is usually supported or we can just pass the whole thing.
        # Note: The SDK's FunctionDeclaration accepts a dict for `parameters`.
        # However, to avoid strict typing issues, we can just build the raw dict representation
        # and let the SDK serialize it.

        raw_func = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        if parameters:
            raw_func["parameters"] = parameters

        tool = types.Tool(function_declarations=[raw_func])
        return tool

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[types.Content]]:
        """
        Convert messages to Gemini format.
        Extracts the first system prompt as `system_instruction`.
        Converts the rest to `Content` objects.
        """
        system_instruction = None
        gemini_contents = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "system":
                if system_instruction is None:
                    system_instruction = content
                else:
                    # Gemini only takes one system instruction, append or ignore if multiple exist.
                    system_instruction += f"\n\n{content}"
                continue

            gemini_role = "user" if role in ("user", "tool") else "model"

            parts = []
            if content:
                # Handle multimodal (from ContextBuilder: list of dicts with type/image_url)
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item["text"]))
                        elif item.get("type") == "image_url":
                            # Base64 string from data URI
                            b64_data = item["image_url"]["url"].split(",", 1)[1]
                            mime_type = (
                                item["image_url"]["url"].split(";", 1)[0].split(":")[1]
                            )
                            import base64

                            raw_bytes = base64.b64decode(b64_data)
                            parts.append(
                                types.Part.from_bytes(
                                    data=raw_bytes, mime_type=mime_type
                                )
                            )
                else:
                    parts.append(types.Part.from_text(text=content))

            # Handle tool calls (model invoking a tool)
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    # Need to parse json args string back to dict for Gemini
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except:
                        args = {}
                    parts.append(
                        types.Part.from_function_call(
                            name=fn.get("name", ""), args=args
                        )
                    )

            # Handle tool responses (user role equivalent providing tool output)
            if role == "tool":
                # Ensure we have a part for the response.
                # If content is string (which it usually is from ToolRegistry), wrap it.
                tool_name = msg.get("name", "tool")
                # Fallback empty object if output is empty
                resp_content = {"output": content} if content else {}
                parts.append(
                    types.Part.from_function_response(
                        name=tool_name, response=resp_content
                    )
                )

            if parts:
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))

        return system_instruction, gemini_contents

    async def stream_chat(
        self, messages: List[Dict[str, Any]], tools_def: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams chat responses compatible with AgentLoop.
        Yields normalized events like `text_delta` and `tool_calls`.
        """
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools_def:
            # We convert each function definition into a Tool.
            gemini_tools = [self._convert_tool_to_gemini(t) for t in tools_def]
            config_kwargs["tools"] = gemini_tools

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            # Wait for generator
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
        except Exception as e:
            yield {"type": "error", "content": f"Gemini API Error: {str(e)}"}
            return

        async for chunk in response_stream:
            # Usage tracking
            if chunk.usage_metadata:
                yield {
                    "type": "token_usage",
                    "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                    "completion_tokens": chunk.usage_metadata.candidates_token_count,
                    "total_tokens": chunk.usage_metadata.total_token_count,
                }

            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]

            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                # Standard Text Delta
                if part.text:
                    yield {"type": "text_delta", "content": part.text}

                # "Thoughts Signature" Text Delta
                if (
                    hasattr(part, "thought") and part.thought
                ):  # Or however the SDK exposes it. Usually it's `part.executable_code` or `part.text` with a specific role, but for the flash-thinking model it's expected to be a distinct field or appended. If it's `executable_code`, it's different. If the SDK natively surfaces `thought` or `thought_process`:
                    yield {"type": "thought_delta", "content": part.thought}
                elif getattr(part, "executable_code", None):
                    yield {
                        "type": "thought_delta",
                        "content": f"```python\n{part.executable_code.code}\n```\n",
                    }
                elif getattr(part, "code_execution_result", None):
                    yield {
                        "type": "thought_delta",
                        "content": f"Execution Result: {part.code_execution_result.output}\n",
                    }

                # Tool Calls
                if part.function_call:
                    # Normalize to OpenAI's tool_calls structure that loop expects
                    fc = part.function_call
                    # genai args are Dict, convert to JSON string
                    args_str = json.dumps(fc.args) if fc.args else "{}"
                    # Generate a mock ID since Gemini doesn't use IDs natively
                    import uuid

                    tc_id = f"call_{uuid.uuid4().hex[:10]}"

                    yield {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": fc.name, "arguments": args_str},
                            }
                        ],
                    }
