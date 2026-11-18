# -*- coding: utf-8 -*-
"""
最小 Agent：两个工具 + 一个循环
================================================================
这个文件的目的不是做一个 Agent 产品，是让你**亲手跑通一次循环**，
从而能回答面试官的问题：「Agent 和 RAG 的区别在哪？」

RAG 是：检索一次 → 生成一次 → 结束。（直线）
Agent 是：想 → 调工具 → 看结果 → 再想 → …… → 决定不再调，回答。（循环）

区别只有两点：① 调哪个工具由模型自己决定；② 可以调多次。

复用你已有的检索代码：search_kb 里就是你的 retrieve()。
所以这一步不是从零开始，是把「检索」包成 Agent 能调用的一个工具。

依赖：openai / numpy / python-dotenv（和 baseline.py 一样，不用 LangChain）

用法：
    python agent_min.py                        # 跑 4 道预设题
    python agent_min.py --ask "宠物饿了怎么办"   # 问单题
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(HERE / ".env")

from openai import OpenAI  # noqa: E402

from baseline_myimpl import build_index, load_chunks, retrieve  # noqa: E402

# ══════════════════════════════════════════════════════════════
# 1. 准备知识库（复用你的检索代码，一行没改）
# ══════════════════════════════════════════════════════════════
CHUNKS = load_chunks()
VOCAB, IDF, MATRIX = build_index(CHUNKS)


# ══════════════════════════════════════════════════════════════
# 2. 两个工具：普通 Python 函数，Agent 能不能用取决于 schema 写得清不清楚
# ══════════════════════════════════════════════════════════════
def search_kb(query: str) -> str:
    """工具一：查知识库。就是你原来的 retrieve()，外面包了一层字符串输出。"""
    hits = retrieve(query, MATRIX, CHUNKS, VOCAB, IDF, k=3)
    if not hits or hits[0][0] <= 0:
        return "知识库中没有相关内容。"
    return "\n".join(
        f"[片段{i}]（{c['chunk_id']}）{c['text'][:150]}" for i, (_, c) in enumerate(hits, 1)
    )


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calc(expression: str) -> str:
    """工具二：四则运算。故意不用 eval()，只允许数字和 + - * / ( )。"""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("只支持数字与 + - * / ( )")

    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"计算失败：{e}"


# ══════════════════════════════════════════════════════════════
# 3. 工具说明书（schema）：模型只看得到这段文字，看不到函数实现
#    描述写得含糊，模型就会选错工具或传错参数 —— 这是 Agent 最常见的故障
# ══════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "查询「单词萌宠」小游戏的资料库，包括玩法、宠物养成规则、金币获取、账号问题。"
                           "只用于回答游戏相关问题，不能用于数学计算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题，用中文完整短句"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "四则运算计算器，支持 + - * / 和括号。凡是需要算数的地方都必须调用它，不要自己心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "如 (3*2)*5"},
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是「单词萌宠」的客服助手。可以调用工具来完成任务。\n"
    "规则：\n"
    "1. 问游戏玩法、规则、金币等资料问题时，必须调用 search_kb，不要凭记忆回答。\n"
    "2. 涉及数字计算时，必须调用 calc，不要自己算。\n"
    "3. 工具结果里没有答案，就回答「资料中未提及」，不要猜测。\n"
    "4. 回答不超过 100 字，面向初中生。"
)

MAX_STEPS = 5  # 防止死循环：Agent 必须有步数上限，这是工程常识


# ══════════════════════════════════════════════════════════════
# 4. 循环：Agent 的全部秘密就在这个 for 里
# ══════════════════════════════════════════════════════════════
def run(client, question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            messages=messages,
            tools=TOOLS,
            temperature=0,
            max_tokens=400,
        )
        msg = resp.choices[0].message

        # 分支一：模型认为不需要再调工具 → 循环结束
        if not msg.tool_calls:
            trace.append(f"第{step}步 结束：直接回答")
            if verbose:
                for line in trace:
                    print("  " + line)
            return msg.content or ""

        # 分支二：模型要求调工具 → 执行，把结果塞回对话，继续下一轮
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "search_kb":
                result = search_kb(args.get("query", ""))
            elif name == "calc":
                result = calc(args.get("expression", ""))
            else:
                result = f"未知工具 {name}"
            trace.append(f"第{step}步 调 {name}({json.dumps(args, ensure_ascii=False)}) → {result[:50]}…")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if verbose:
        for line in trace:
            print("  " + line)
    return f"超过 {MAX_STEPS} 步仍未给出答案，已停止（防止死循环）。"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None)
    a = ap.parse_args()

    c = OpenAI(
        api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
        timeout=float(os.getenv("TIMEOUT", "60")),
    )

    qs = [a.ask] if a.ask else [
        "宠物饿了怎么办？",                              # 应调 search_kb
        "每只宠物每天掉 2 点饥饿值，3 只养 5 天共掉多少点？",  # 应先 calc（可能再查 kb）
        "宠物死了怎么办",                                # 应调 search_kb
        "今天北京天气怎么样？",                            # 应拒答，不硬编
    ]

    for q in qs:
        print("=" * 70)
        print(f"问题：{q}")
        print(f"回答：{run(c, q)}")
