import os
import yaml
import re


class SkillsLoader:
    """
    技能加载器模块。
    用于从指定目录加载 Markdown 格式的技能文件 (通常是 SKILL.md)。
    文件中可以包含 YAML frontmatter (包含技能元数据，如描述和状态) 及正文（给 Agent 参考的提示词）。
    """

    def __init__(self, workspace: str):
        self.skills_dir = os.path.join(workspace, "skills")
        self.skills = []
        self.load_all_skills()

    def load_all_skills(self):
        """遍历目录并解析出所有可用的技能"""
        self.skills = []
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)

        for root, dirs, files in os.walk(self.skills_dir):
            for file in files:
                if file == "SKILL.md":
                    skill_name = os.path.basename(root)
                    skill_path = os.path.join(root, file)
                    meta, content = self._parse_markdown_with_frontmatter(skill_path)

                    self.skills.append(
                        {
                            "name": skill_name,
                            "description": meta.get("description", "无描述"),
                            "active": meta.get("active", True),  # 默认激活
                            "always_load": meta.get(
                                "always_load", False
                            ),  # 是否强制全局加载
                            "path": skill_path.replace("\\", "/"),
                            "content": content,
                        }
                    )

    def _parse_markdown_with_frontmatter(self, filepath: str):
        """解析 Markdown 文件，拆分出 yaml header 和剩余内容"""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 匹配首部的 --- YAML内容 ---
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if match:
            yaml_content = match.group(1)
            markdown_content = match.group(2)
            try:
                meta = yaml.safe_load(yaml_content) or {}
                return meta, markdown_content.strip()
            except Exception:
                return {}, text.strip()

        return {}, text.strip()

    def get_always_skills_prompt(self) -> str:
        """
        获取需要“始终自动加载”的技能 (always_load: true) 的全部 Prompt 内容。
        """
        always_skills = [
            s
            for s in self.skills
            if s.get("active", True) and s.get("always_load", False)
        ]
        if not always_skills:
            return ""

        prompt_parts = ["# 常驻核心技能 (Always-loaded Skills)"]
        prompt_parts.append("你目前具备以下常驻核心技能，你可以随时使用它们：\n")

        for skill in always_skills:
            prompt_parts.append(f"## 技能：{skill['name']}")
            prompt_parts.append(f"{skill['content']}\n")

        return "\n".join(prompt_parts)

    def build_skills_summary_prompt(self) -> str:
        """
        构建可选技能列表的摘要信息，供 Agent 阅读并决定是否需要使用 `read_file` 工具查看详情。
        """
        available_skills = [
            s
            for s in self.skills
            if s.get("active", True) and not s.get("always_load", False)
        ]
        if not available_skills:
            return ""

        prompt_parts = ["# 可选扩展技能 (Available Skills)"]
        prompt_parts.append(
            "以下技能扩展了你的能力。想使用某项技能前，请务必使用 `read_file` 工具读取相应路径下的 SKILL.md 文件学习具体用法。\n"
        )

        for skill in available_skills:
            prompt_parts.append(f"- **{skill['name']}**: {skill['description']}")
            prompt_parts.append(f"  > 技能指南文件路径：`{skill['path']}`")

        return "\n".join(prompt_parts)

    def get_skills_summary(self) -> list:
        """为前端状态展示提取所有的技能摘要列表"""
        return [
            {"name": s["name"], "description": s["description"], "active": s["active"]}
            for s in self.skills
        ]
