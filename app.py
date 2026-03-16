import json
import logging
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel

import os
import yaml
from core.agent import TinyAgent

# 加载配置文件
config_path = "config.yaml"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

llm_config = config.get("llm", {})

workspace_path = "./workspace"
outputs_path = os.path.join(workspace_path, "outputs")
os.makedirs(outputs_path, exist_ok=True)

# 动态配置 LLM 提供商
provider_name = llm_config.get("provider", "openai")
# 如果没有独立写 provider 的 dict, 就提供整个 llm_config 过去回退兼容
provider_config = llm_config.get(provider_name, llm_config)
# Fallback model parsing since it might be defined top-level in llm_config for simple setups
if "model" not in provider_config:
    provider_config["model"] = llm_config.get("model", "gpt-4o-mini")

agent = TinyAgent(
    workspace_dir=workspace_path,
    provider=provider_name,
    provider_config=provider_config,
)

app = FastAPI(title="Tiny Agent Backend")

# 挂载静态资源
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory=outputs_path), name="outputs")


@app.get("/")
async def root():
    """返回前端主页"""
    return FileResponse("static/index.html")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    流式对话接口。使用 GET / POST 无所谓，这里为了获取 query 用 POST 接收 message 后，
    将其转换成 SSE (Server-Sent Events) 返回。
    """

    async def sse_generator() -> AsyncGenerator[str, None]:
        # 遍历 agent_loop 的每一个步骤触发的字典事件
        async for event in agent.chat_stream(req.message):
            # 将 python 字典格式化为 JSON 字符串
            data_str = json.dumps(event, ensure_ascii=False)
            # SSE 要求格式以 data: 开头，以 \n\n 结尾
            yield f"data: {data_str}\n\n"

    # 指定媒体类型为 text/event-stream 这是 SSE 标准的配置
    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.get("/api/status")
async def get_status():
    """获取侧边栏展示的相关状态（刷新并返回技能和支持的工具）"""
    agent.skills.load_all_skills()  # Dynamic reload
    return {"skills": agent.get_skills_summary(), "tools": agent.get_tools_summary()}


@app.get("/api/memory")
async def get_memory():
    """获取当前 agent 的上下文和长期记忆"""
    messages = agent.memory.get_messages(window_size=20)
    system_prompt = agent.context.build_system_prompt()
    long_term_memory = agent.memory.get_long_term_memory()

    # 统计信息
    stats = {
        "total_messages_in_window": len(messages),
        "has_long_term_memory": bool(long_term_memory),
    }

    return {
        "stats": stats,
        "long_term_memory": long_term_memory,
    }


@app.get("/api/history")
async def get_history():
    """获取完整的历史会话和累积 token 消耗用于前端恢复渲染"""
    return {"messages": agent.memory.messages, "tokens": agent.memory.get_tokens()}


@app.get("/api/outputs")
async def list_outputs():
    """获取工作区所有的输出文件列表"""
    files = []
    if os.path.exists(outputs_path):
        for f in os.listdir(outputs_path):
            file_path = os.path.join(outputs_path, f)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({"name": f, "size": stat.st_size, "mtime": stat.st_mtime})
        # 按修改时间倒序（最新的在前面）
        files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"files": files}


@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """Delete a specific file from the workspace outputs directory"""
    # Security: Prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return {"status": "error", "message": "Invalid filename"}

    file_path = os.path.join(outputs_path, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": f"Deleted {filename}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": "File not found"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到 workspace outputs 目录"""
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path, exist_ok=True)
    file_path = os.path.join(outputs_path, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/clear")
async def clear_memory():
    """清理内存会话记录"""
    agent.clear_memory()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logging.info("Starting Tiny Agent server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
