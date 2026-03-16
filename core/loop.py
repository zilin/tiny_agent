import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional

from .tools import ToolRegistry

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentLoop:
    """
    事件循环模块。
    核心职责：
    1. 调用通用 LLMProvider 获取流式返回
    2. 判断是返回纯文本还是触发 Tool Call
    3. 解析 Tool Call 并调度 `ToolRegistry` 执行
    4. 将工具的运行结果再反向注入上下文请求大模型（循环直到结束）
    5. 通过 Async Generator 把整个过程的状态透传给外部（用于 SSE 推送和可视化）
    """

    def __init__(self, provider, tool_registry: ToolRegistry):
        self.provider = provider
        self.tool_registry = tool_registry

    async def run(
        self, messages: list[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        启动与大模型的交互。
        产生字典形式的事件：
        {
          "type": "text_delta" | "tool_call_start" | "tool_call_end" | "token_usage" | "thought_delta",
          "content": "", ...
        }
        """
        current_messages = list(messages)
        max_iterations = 10  # 防止无限死循环调用工具的防御阈值
        iteration = 0
        tools_def = self.tool_registry.get_definitions()

        while iteration < max_iterations:
            iteration += 1

            # 清理消息历史中的非法字段（如空 tool_calls），兼容各路 API 后端
            cleaned_messages = []
            for m in current_messages:
                new_m = m.copy()
                if "tool_calls" in new_m and not new_m["tool_calls"]:
                    del new_m["tool_calls"]
                if "content" in new_m and new_m["content"] == "":
                    new_m["content"] = None
                cleaned_messages.append(new_m)

            logger.info(
                f"[AgentLoop] 发起第 {iteration} 轮请求，携带 {len(cleaned_messages)} 条历史"
            )

            # The provider now yields generic dictionaries:
            # text_delta, thought_delta, tool_calls (fully buffered list), token_usage, error
            assistant_msg = {"role": "assistant", "content": ""}

            tool_calls = []

            async for event in self.provider.stream_chat(cleaned_messages, tools_def):
                event_type = event.get("type")

                if event_type == "text_delta":
                    assistant_msg["content"] += event["content"]
                    yield event

                elif event_type == "thought_delta":
                    yield event

                elif event_type == "token_usage" or event_type == "error":
                    yield event

                elif event_type == "tool_calls":
                    tool_calls = event["tool_calls"]

            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            # 处理空内容，防止某些后端报错
            if not assistant_msg["content"]:
                assistant_msg["content"] = None

            current_messages.append(assistant_msg)

            if not tool_calls:
                break

            # 处理每个工具的调用结果
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args_str = tc["function"]["arguments"]

                yield {
                    "type": "tool_call_start",
                    "id": tc["id"],
                    "name": tool_name,
                    "arguments": tool_args_str,
                }

                logger.info(f"执行工具 '{tool_name}' 参数: {tool_args_str}")

                result = await self.tool_registry.execute(tool_name, tool_args_str)
                logger.info(f"执行结果: {result[:100]}...")

                yield {
                    "type": "tool_call_end",
                    "id": tc["id"],
                    "name": tool_name,
                    "result_summary": result[:100]
                    + ("..." if len(result) > 100 else ""),
                }

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": result,
                    }
                )

        yield {"type": "turn_end", "new_messages": current_messages[len(messages) :]}
