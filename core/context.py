import time
import platform
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional

from .memory import MemoryStore
from .skills import SkillsLoader


class ContextBuilder:
    """
    上下文构建器模块。
    按顺序组装并构建完整的发送给大模型的 Message Payload:
    1. System Prompt (包含核心设定，时间，可用技能，长记忆等)
    2. 历史对话 (Short-Term Memory)
    3. User 的新一条输入
    """

    def __init__(
        self, memory_store: MemoryStore, skills_loader: SkillsLoader, workspace_dir: str
    ):
        self.memory = memory_store
        self.skills = skills_loader
        self.workspace_dir = workspace_dir

    def build_system_prompt(self) -> str:
        """构建系统提示词"""
        parts = []

        # 1. 基础人格与时间设定
        parts.append(self._get_identity())

        # 2. 挂载常驻核心技能 (always-loaded skills)
        always_skills_prompt = self.skills.get_always_skills_prompt()
        if always_skills_prompt:
            parts.append(always_skills_prompt)

        # 3. 挂载长期记忆 (如果非空)
        long_term_fact = self.memory.get_long_term_memory()
        if long_term_fact:
            parts.append(f"# 工作记忆和参考事实\n\n{long_term_fact}")

        # 4. 附加可选技能列表清单
        skills_summary = self.skills.build_skills_summary_prompt()
        if skills_summary:
            parts.append(skills_summary)

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """获取核心人格设定和运行环境信息"""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        tz = time.strftime("%Z") or "UTC"
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        workspace_path = Path(self.workspace_dir).resolve().as_posix()

        return f"""你名叫 tinybot，是一个有用的 AI 助手。 

## 当前时间
{now} ({tz})

## 运行环境
{runtime}

## 工作区
你的工作区位于: {workspace_path}
- 长期记忆: {workspace_path}/memory/MEMORY.md
- 历史日志: {workspace_path}/memory/HISTORY.md (支持 grep 搜索)
- 输出目录: {workspace_path}/outputs/
- 自定义技能: {workspace_path}/skills/{{skill-name}}/SKILL.md

> [!IMPORTANT]
> **绝对强制要求：** 除了读取记忆（`memory/`）和读取技能配置（`skills/`）外，无论是文档、代码、图片、音频、测试文件还是任何工具的执行生成产物，**只要你需要新建或修改目标文件存放结果，你《必须》将它们统统存放在 `{workspace_path}/outputs/` 目录内**。
> **绝不允许**在使用文件操作工具时将其直接放在 `{workspace_path}` 根目录或其他未授权位置！如果不指定具体完整路径，请自行在文件名前加上 `{workspace_path}/outputs/`。

直接使用文本回复对话。仅在需要发送到特定聊天频道时使用 'message' 工具。

## 工具调用指南
- 在调用工具之前，你可以简要说明你的意图（例如“让我检查一下”），但绝不要在收到结果之前预测或描述预期的结果。
- 不要假设文件或目录存在 — 使用 read_file 或 exec (ls) 来验证。
- 在使用 edit_file 或 write_file 修改文件之前，请先阅读以确认其当前内容。
- 在写入或编辑文件后，如果准确性很重要，请重新阅读它。
- 如果工具调用失败，请在尝试不同方法之前分析错误。

## 记忆
- 记住重要的事实：写入 {workspace_path}/memory/MEMORY.md
- 回忆过去的事件：使用 grep 搜索 {workspace_path}/memory/HISTORY.md"""

    def build_messages(
        self, current_user_message: str, media: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        组装全部消息用于大模型 API 调用
        """
        messages = []

        # 第一条永远是 System Message
        system_prompt = self.build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 过去的所有历史 (如果有，且在限制窗口内)
        # 注意: tools 和 tool result 也会保存在 history 里，用于大模型判断前置状态
        history_msgs = self.memory.get_messages(window_size=20)
        messages.extend(history_msgs)

        # 当前轮次用户的输入，追加到 Payload 末尾
        user_content = self._build_user_content(current_user_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: Optional[List[str]]) -> Any:
        """构建包含可选图片的 User Message Content"""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: List[Dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> List[Dict[str, Any]]:
        """
        辅助方法：向消息列表中添加工具执行结果
        """
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages

    def add_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        辅助方法：向消息列表中添加助手的回复
        """
        msg: Dict[str, Any] = {"role": "assistant"}

        msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)
        return messages
