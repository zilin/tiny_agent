import os
import json
import asyncio
import re
from typing import Dict, Any, Callable, List


class BaseTool:
    """
    基础工具类，所有自定义工具都需要继承此类。
    提供了工具的名称、描述、参数结构等大模型需要的元数据。
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_openai_function(self) -> Dict[str, Any]:
        """将工具转换为 OpenAI API 兼容的 function 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, **kwargs) -> str:
        """执行工具的具体逻辑，子类必须实现"""
        raise NotImplementedError("子类必须实现 execute 方法")


class ReadFileTool(BaseTool):
    """读取文件工具"""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="读取指定文件的内容。注意，如果文件太大可能会截断或报错。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件的绝对或相对路径",
                    }
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # 防止单次读取文件过大，限制前10000个字符
                if len(content) > 10000:
                    return content[:10000] + "\n...[文件内容过长被截断]"
                return content
        except Exception as e:
            return f"读取文件失败: {str(e)}"


class WriteFileTool(BaseTool):
    """写入文件工具"""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="将内容写入到指定文件中。如果文件不存在则会创建，如果存在则会覆盖。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要写入的文件的绝对或相对路径",
                    },
                    "content": {"type": "string", "description": "要写入的内容文本"},
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"成功写入文件: {path}"
        except Exception as e:
            return f"写入文件失败: {str(e)}"


class EditFileTool(BaseTool):
    """编辑文件工具 (简单查找替换)"""

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="编辑指定文件的内容。通过查找旧字符串并替换为新字符串。建议先读取文件内容确认。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要编辑的文件的绝对或相对路径",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "要被替换的原始文本字符串",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "替换后的新文本字符串",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        )

    async def execute(self, path: str, old_str: str, new_str: str) -> str:
        try:
            if not os.path.exists(path):
                return f"错误：文件 {path} 不存在"

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_str not in content:
                return f"错误：在文件内容中未找到指定的 old_str"

            new_content = content.replace(old_str, new_str)

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"成功编辑文件: {path}"
        except Exception as e:
            return f"编辑文件失败: {str(e)}"


class ShellTool(BaseTool):
    """
    执行 Shell 命令工具。
    提供执行系统命令的能力，配有超时与高危命令拦截以维护安全。
    """

    def __init__(self, timeout: int = 60):
        super().__init__(
            name="exec",
            description="执行 Shell 命令并返回输出。谨慎使用。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 Shell 命令"},
                    "working_dir": {"type": "string", "description": "可选的执行目录"},
                },
                "required": ["command"],
            },
        )
        self.timeout = timeout
        # 拦截常见高危操作
        self.deny_patterns = [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"(?:^|[;&|]\s*)format\b",  # format
            r"\b(mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]

    async def execute(self, command: str, working_dir: str = None) -> str:
        cwd = working_dir or os.getcwd()
        guard_error = self._guard_command(command)
        if guard_error:
            return guard_error

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"错误：命令执行超时（超过 {self.timeout} 秒）"

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\n退出状态码: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(无输出)"

            if len(result) > 10000:
                result = (
                    result[:10000] + f"\n... (截断，剩余 {len(result) - 10000} 个字符)"
                )

            return result
        except Exception as e:
            return f"执行命令时发生异常: {str(e)}"

    def _guard_command(self, command: str) -> str | None:
        cmd = command.strip()
        lower = cmd.lower()
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "错误: 命令被安全策略拦截 (检测到危险模式)"
        return None


class ToolRegistry:
    """工具注册中心，负责管理和执行所有工具"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        # 默认注册基础的文件操作工具
        self.register(ReadFileTool())
        self.register(WriteFileTool())
        self.register(EditFileTool())
        # 注册 Shell 工具
        self.register(ShellTool())

    def register(self, tool: BaseTool):
        """注册一个新工具"""
        self.tools[tool.name] = tool

    def get_definitions(self) -> List[Dict[str, Any]]:
        """获取所有已注册工具的 OpenAI function 定义列表"""
        return [tool.to_openai_function() for tool in self.tools.values()]

    async def execute(self, name: str, arguments_json: str) -> str:
        """
        根据工具名称和 JSON 格式的参数执行对应的工具
        """
        if name not in self.tools:
            return f"错误：未找到名为 '{name}' 的工具"

        tool = self.tools[name]
        try:
            kwargs = json.loads(arguments_json)
            result = await tool.execute(**kwargs)
            return str(result)
        except json.JSONDecodeError:
            return "错误：提供的参数不是有效的 JSON 格式"
        except Exception as e:
            return f"执行工具 '{name}' 时发生异常: {str(e)}"
