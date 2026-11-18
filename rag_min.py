# -*- coding: utf-8 -*-
"""
最小可跑 RAG：切块 → 检索 → 拼 prompt → 生成，约 130 行
================================================================
依赖只有 openai + numpy（+ dotenv 读配置），刻意不用 LangChain。

六步链路，每一步都打印中间结果，便于定位「一次问答里有几次模型调用、
几次向量计算、分别在哪一步」。

用法（干净环境）：
    python rag_min.py "你的问题"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

TOP_K = int(os.getenv("TOP_K", "3"))

# ============ 0. 语料（先用内存里的三小段，完整版见 baseline.py）============
DOCS = {
    "宠物.md": "单词萌宠是一款背单词小游戏。学生答对得金币，金币用来买种子和菜谱。"
              "错词会进入复习队列。复习间隔共 8 档，从 5 分钟到 15 天逐步拉长。",
    "账号.md": "微信小游戏个人账号无法配置业务域名。没有业务域名就不能在游戏内直接请求外部接口。"
              "解决办法是把后端挂在云开发上，或把逻辑全部放到前端本地实现。",
    "上线.md": "小游戏上线需要先上传代码、生成体验版，再提交审核。审核需要软件著作权证书。"
              "从提交到拿到证书通常要一到两个月，赶时间时应优先做 H5 网页版。",
}

# ============ 1. 切分：按中文句末标点切成两句一块 ============
def split_chunks(text: str, size: int = 2) -> list[str]:
    import re
    sents = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    return ["".join(sents[i:i + size]) for i in range(0, len(sents), size)]

chunks, sources = [], []
for name, text in DOCS.items():
    for c in split_chunks(text):
        chunks.append(c)
        sources.append(name)

# ============ 2. 向量化：字符 bigram 词袋（替代 bge，先看懂余弦）============
def bigrams(s: str) -> list[str]:
    s = "".join(s.split())
    return [s[i:i + 2] for i in range(len(s) - 1)]

VOCAB = {g: i for i, g in enumerate(sorted({g for c in chunks for g in bigrams(c)}))}

def embed(s: str) -> np.ndarray:
    v = np.zeros(len(VOCAB), dtype=np.float32)
    for g in bigrams(s):
        if g in VOCAB:
            v[VOCAB[g]] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v            # L2 归一化，之后点积就等于余弦

matrix = np.vstack([embed(c) for c in chunks])   # 向量库：一个 N×D 矩阵

# ============ 3. 检索：一次矩阵乘法 = 全库余弦相似度 ============
def retrieve(question: str, k: int) -> list[tuple[float, str, str]]:
    scores = matrix @ embed(question)            # 向量计算只有这一行
    idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i], sources[i]) for i in idx]


# ============ 4. 拼 prompt + 调模型 ============
def answer(question: str, k: int = TOP_K) -> None:
    hits = retrieve(question, k)

    print("=" * 68)
    print(f"问题：{question}    top_k={k}")
    print("-" * 68)
    for rank, (score, chunk, src) in enumerate(hits, 1):
        print(f"  [{rank}] 余弦={score:.3f}  来源={src}  {chunk[:40]}…")

    context = "\n".join(f"[片段{i}] {c}" for i, (_, c, _) in enumerate(hits, 1))

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        print("\n[提示] 未配置 OPENAI_API_KEY，检索部分已完成，跳过生成。")
        return

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "只能依据资料回答；资料没有就答「资料中未提及」。"},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
        temperature=0,
        max_tokens=300,
    )
    print("\n" + "=" * 68)
    print("回答：", resp.choices[0].message.content)
    print("-" * 68)
    print(f"本次消耗 prompt_tokens={resp.usage.prompt_tokens} "
          f"completion_tokens={resp.usage.completion_tokens}")
    print("模型调用 1 次；向量计算 1 次（matrix @ q，即 N 个余弦）；"
          f"向量库共 {len(chunks)} 段。")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "个人账号为什么不能直接请求外部接口？"
    if "--topk" in sys.argv:
        answer(q, int(sys.argv[sys.argv.index("--topk") + 1]))
    else:
        answer(q)
