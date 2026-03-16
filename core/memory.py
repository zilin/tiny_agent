import json
import os
from typing import List, Dict, Any


class MemoryStore:
    """
    轻量级的记忆存储模块，支持对话历史的持久化。
    包含：
      1. message history: 轮次的对话（系统、用户、助手消息）。
      2. long term memory: 一些持久存在的事实（可选的高级功能）。
    """

    def __init__(self, workspace_dir: str, session_id: str = "default"):
        self.memory_dir = os.path.join(workspace_dir, "memory")
        os.makedirs(self.memory_dir, exist_ok=True)

        self.history_file = os.path.join(self.memory_dir, f"{session_id}_history.json")
        self.tokens_file = os.path.join(self.memory_dir, f"{session_id}_tokens.json")
        self.long_term_file = os.path.join(self.memory_dir, "MEMORY.md")

        # 恢复状态
        self.messages: List[Dict[str, Any]] = self._load_history()
        self.tokens: Dict[str, int] = self._load_tokens()

    def _load_tokens(self) -> Dict[str, int]:
        """从文件中加载已消耗的 token 统计"""
        default_tokens = {"prompt": 0, "completion": 0}
        if os.path.exists(self.tokens_file):
            try:
                with open(self.tokens_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        "prompt": data.get("prompt", 0),
                        "completion": data.get("completion", 0),
                    }
            except Exception:
                return default_tokens
        return default_tokens

    def _save_tokens(self):
        """保存 token 消耗记录"""
        with open(self.tokens_file, "w", encoding="utf-8") as f:
            json.dump(self.tokens, f, ensure_ascii=False, indent=2)

    def add_tokens(self, prompt_tokens: int, completion_tokens: int):
        """累加 token 消耗"""
        self.tokens["prompt"] += prompt_tokens
        self.tokens["completion"] += completion_tokens
        self._save_tokens()

    def get_tokens(self) -> Dict[str, int]:
        """获取当前累加的 token 消耗"""
        return self.tokens

    def _load_history(self) -> List[Dict[str, Any]]:
        """从文件中加载对话记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        """保存对话记录到 JSON 文件"""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def add_message(self, message: Dict[str, Any]):
        """
        新增一条消息到短期历史中并持久化。
        消息角色通常是 "system", "user", "assistant" 或者是 "tool"
        """
        self.messages.append(message)
        self._save_history()

    def get_messages(self, window_size: int = 20) -> List[Dict[str, Any]]:
        """
        获取对话历史。为了避免超出 token 限制，可以通过 window_size 截断早期的部分。
        注意：截断不能破坏大模型的连续性要求。例如，如果包含 tool_calls，则必须包含对应的 tool response 消息。
        如果简单按数量截取会导致 orphaned tool_calls，这会引发 API 错误。
        这里我们从后向前遍历找到一个安全的切片点（比如用户最新一轮发起对话的地方，或者确保工具链完整的起始点）。
        """
        if len(self.messages) <= window_size:
            return self.messages

        # 安全截断算法：优先获取最后的 window_size 条消息
        candidate = self.messages[-window_size:]

        # 从该 candidate 第一个元素向后检查是否有无头 tool 消息，或未闭合的 tool_calls
        # 如果 candidate 的第一条消息是个 tool (意味着它的 parent tool call 被截掉了)，
        # 我们就得继续往左找，直到找到发起这个 tool call 的 assistant 消息，甚至更早的 user 消息。

        # 为了简单且有效，我们可以直接寻找倒数第 window_size 个消息之前的最近的一个 user 消息作为起点。
        # 这样能保证这是一个完整的针对用户问题的交互序列。
        start_idx = max(0, len(self.messages) - window_size)

        # 如果起点恰好切断了逻辑链条
        # 寻找最近的一个非 tool，非带有 tool_calls 且能自圆其说的起点 (最好是 user)
        # 向前找，直到找到 role == "user"
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        return self.messages[start_idx:]

    def get_long_term_memory(self) -> str:
        """读取持久状态的长记忆（如果存在的话）"""
        if os.path.exists(self.long_term_file):
            try:
                with open(self.long_term_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def save_long_term_memory(self, memory_text: str):
        """保存归纳后的长记忆（供其它 Agent 或任务调用）"""
        with open(self.long_term_file, "w", encoding="utf-8") as f:
            f.write(memory_text)

    def clear_history(self):
        """清空当前会话的对话记录及 token 记录"""
        self.messages = []
        self._save_history()
        self.tokens = {"prompt": 0, "completion": 0}
        self._save_tokens()
