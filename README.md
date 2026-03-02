<p align="center">
  <img src="./logo.jpg" width="120" alt="TinyAgent Logo" />
</p>

<h1 align="center">TinyAgent</h1>

<p align="center">
  <strong>6 个文件，1 个能用的 AI Agent</strong><br/>
  用最少的代码理解 Agent 的本质
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/framework-FastAPI-009688?style=flat-square&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LLM-OpenAI%20Compatible-412991?style=flat-square&logo=openai" alt="LLM" />
  <img src="https://img.shields.io/badge/lines-~700-green?style=flat-square" alt="Lines" />
</p>

---

## ✨ 什么是 TinyAgent

TinyAgent 是一个 **最小可运行的 AI Agent 实现**。它不是框架，而是一份可以直接跑起来的教学级源码——用 6 个 Python 文件、不到 700 行代码，完整实现了：

- 🔄 **流式对话** — SSE 实时推送，逐字输出
- 🛠️ **Tool Calling** — 多轮工具调用 + 防死循环保险丝
- 🧠 **对话记忆** — 短期历史窗口 + 长期 Markdown 记忆
- 📦 **技能插件** — 放一个 `SKILL.md` 文件即装即用
- 🔒 **安全防护** — Shell 高危命令正则拦截

## 🏗️ 项目结构

```
tiny_agent/
├── app.py              # FastAPI 入口，HTTP 路由 + SSE
├── config.yaml         # LLM 配置（API Key、模型、地址）
├── requirements.txt    # Python 依赖
├── static/             # 前端静态资源
├── workspace/          # 运行时工作区
│   ├── memory/         #   对话记忆存储
│   ├── skills/         #   技能插件目录
│   └── outputs/        #   文件输出目录
└── core/               # 核心模块
    ├── agent.py        #   总指挥，组装所有组件
    ├── loop.py         #   事件循环，流式 Tool Calling
    ├── tools.py        #   工具注册与执行
    ├── memory.py       #   短期 + 长期记忆管理
    ├── skills.py       #   Markdown 技能加载器
    └── context.py      #   上下文 & 系统提示词组装
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd tiny_agent
pip install -r requirements.txt
```

### 2. 配置 LLM

编辑 `config.yaml`，填入你的 API 信息：

```yaml
llm:
  api_key: "your-api-key-here"
  model: "gpt-4o-mini"           # 或 qwen-plus, deepseek-chat 等
  base_url: "https://api.openai.com/v1"  # 兼容 OpenAI 格式的地址
```

> 支持所有 OpenAI 兼容 API：OpenAI、通义千问、DeepSeek、智谱 GLM 等。

### 3. 启动服务

```bash
python app.py
```

访问 `http://localhost:8000` 即可开始对话。

## 🧩 核心架构

```
用户消息 → ContextBuilder 组装上下文 → AgentLoop 调用 LLM
                                          ↓
                                    模型返回文本？→ 流式输出给用户
                                    模型要用工具？→ ToolRegistry 执行
                                          ↓
                                    结果注入上下文 → 再次调用 LLM
                                          ↓
                                    循环直到模型说完 → MemoryStore 保存
```

**核心设计理念：**

| 原则 | 实现 |
|------|------|
| 组合优于继承 | TinyAgent 组合 5 个独立模块，无基类继承 |
| 流式优先 | 全链路 AsyncGenerator，用户秒看到第一个字 |
| 安全兜底 | `max_iterations=10` 防死循环 + Shell 命令黑名单 |
| 零配置扩展 | 扔一个 `SKILL.md` 进 `skills/` 目录即生效 |

## 🛠️ 内置工具

| 工具 | 说明 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入/创建文件 |
| `list_dir` | 列出目录结构 |
| `exec` | 执行 Shell 命令（带安全拦截） |

## 📝 技能系统

在 `workspace/skills/` 下创建目录，放入 `SKILL.md` 即可：

```yaml
---
name: data-analysis
description: 数据分析技能，支持 CSV 处理和可视化
always_load: false
---
# 数据分析技能

当用户需要分析数据时，按以下步骤操作……
```

- `always_load: true` → 全文注入系统提示词（核心技能）
- `always_load: false` → 仅索引名称和描述，按需加载（省 token）

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 流式对话（SSE） |
| `GET` | `/api/status` | 获取技能和工具列表 |
| `GET` | `/api/memory` | 查看记忆状态 |
| `GET` | `/api/history` | 获取完整对话历史 |
| `GET` | `/api/outputs` | 列出输出文件 |
| `POST` | `/api/upload` | 上传文件到工作区 |
| `POST` | `/api/clear` | 清空对话记忆 |
| `DELETE` | `/api/outputs/{filename}` | 删除输出文件 |

## 🤔 为什么不用 LangChain

| | TinyAgent | LangChain |
|--|-----------|-----------|
| 代码量 | ~700 行 | 数万行 |
| 依赖 | 6 个包 | 40+ 个包 |
| 学习曲线 | 读完 6 个文件 | 翻文档几天 |
| 定位 | 教学 + 轻量生产 | 企业级框架 |

TinyAgent 不是要替代任何框架。它的目标是让你 **读完代码后真正理解 Agent 是怎么工作的**。


<p align="center">
  <em>真正重要的东西，往往简单到让人不敢相信。</em>
</p>
