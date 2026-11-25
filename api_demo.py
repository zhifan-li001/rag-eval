"""
知识库问答 HTTP 接口（FastAPI）。

将 baseline_myimpl.py 中的 retrieve() 封装为可远程调用的服务，
包含请求参数校验、上下文长度限制、检索空结果处理与模型异常分层。

启动：
    python -m uvicorn api_demo:app --port 8010

调用：
    curl -X POST http://127.0.0.1:8010/ask -H "Content-Type: application/json" -d "{\"question\":\"宠物饿了怎么办\"}"

浏览器打开 http://127.0.0.1:8010/docs 可以看到自动生成的接口文档。
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from baseline_myimpl import load_chunks, build_index, retrieve

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

# ── 检索索引在进程启动时建一次，不要每个请求都重建 ────────────────
CHUNKS = load_chunks()
VOCAB, IDF, MATRIX = build_index(CHUNKS)

# 分数低于这个值就认为「没检索到相关内容」，不调模型，直接拒答。
# 0.05 是经验阈值，因为向量已 L2 归一化，无关块的余弦分通常在 0.0–0.03。
MIN_SCORE = 0.05

# 一次提问最多允许的上下文字数，防止 topk 开太大把 prompt 撑爆
MAX_CONTEXT_CHARS = 3000

SYSTEM_PROMPT = (
    "你是「单词萌宠」的客服助手。只能依据给定资料回答，不许编造资料里没有的内容。\n"
    "资料里没有答案时，直接回答「资料中未提及」，不要猜测。\n"
    "回答用中文，控制在 150 字以内，尽量口语化，面向初中生。\n"
    "回答末尾用括号注明引用的片段编号，例如（依据片段2）。"
)

app = FastAPI(title="单词萌宠 · 知识库问答 API", version="1.0")


# ── 1. 请求 / 响应模型：这就是「接口契约」 ────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户问题")
    topk: int = Field(5, ge=1, le=20, description="检索返回的片段数，默认 5")


class Chunk(BaseModel):
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    latency_ms: int


# ── 2. 模型客户端（只建一次）─────────────────────────────────────
def _client():
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or not key.isascii() or len(key) < 20:
        return None
    from openai import OpenAI

    return OpenAI(
        api_key=key,
        base_url=os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
        timeout=float(os.getenv("TIMEOUT", "30")),
    )


CLIENT = _client()


# ── 3. 核心接口 ────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()

    # 步骤一：检索
    hits = retrieve(req.question, MATRIX, CHUNKS, VOCAB, IDF, k=req.topk)

    # 步骤二：判断有没有检索到内容。
    # 【关键】这不是错误，是业务正常态。返回 200 + 拒答，不是 4xx/5xx。
    if not hits or hits[0][0] < MIN_SCORE:
        return AskResponse(
            answer="资料中未提及",
            chunks=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # 步骤三：拼上下文，长度超限就截断
    lines, used = [], 0
    for i, (_, c) in enumerate(hits, 1):
        line = f"[片段{i}]（{c['chunk_id']}）{c['text']}"
        if used + len(line) > MAX_CONTEXT_CHARS:
            break
        lines.append(line)
        used += len(line)
    context = "\n".join(lines)

    # 步骤四：调模型。三种失败分别映射成不同状态码。
    if CLIENT is None:
        # 没配 Key 时降级：只返回检索结果，让人知道服务本身是通的
        return AskResponse(
            answer="（未配置模型 Key，仅返回检索结果）",
            chunks=[Chunk(chunk_id=c["chunk_id"], score=round(s, 4), text=c["text"])
                    for s, c in hits],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        resp = CLIENT.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{req.question}"},
            ],
            temperature=0,
            max_tokens=400,
        )
    except Exception as e:  # 超时 → 504；其它上游错误 → 502
        name, msg = type(e).__name__, str(e)
        if "Timeout" in name or "timeout" in msg.lower():
            raise HTTPException(status_code=504, detail="上游模型超时，请重试一次")
        if "RateLimit" in name or "429" in msg:
            raise HTTPException(status_code=429, detail="上游限流，稍后重试")
        # 兜底：不把原始堆栈和 Key 信息抛给调用方
        print(f"[上游错误] {name}: {msg[:200]}")
        raise HTTPException(status_code=502, detail="上游模型异常")

    answer = resp.choices[0].message.content or ""
    return AskResponse(
        answer=answer,
        chunks=[Chunk(chunk_id=c["chunk_id"], score=round(s, 4), text=c["text"])
                for s, c in hits],
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ── 4. 健康检查：运维要能判断「服务活着但依赖挂了」──────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(CHUNKS),
        "vocab": len(VOCAB),
        "llm_configured": CLIENT is not None,
    }
