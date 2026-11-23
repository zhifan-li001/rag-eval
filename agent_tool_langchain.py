"""
用 @tool 装饰器写的同一版 Agent —— 对照 agent_min.py 阅读。

两版的循环逻辑完全一样，只有两处不同：
  1. 工具定义：@tool 读函数签名 + docstring 自动生成 schema，
     替代了手写的 TOOLS 列表（原 94–124 行）。
  2. 调用分发：按名字查表后 .invoke(args)，替代了 if/elif 分支（原 174–179 行）。

注意装饰器没有替你做第三步——把结果塞回对话仍要自己写（本文件第 96 行）。

依赖：pip install langchain-core langchain-openai
配置：与 agent_min.py 同一个 .env（OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL）

运行：
    python agent_tool_langchain.py --ask "宠物饿了怎么办？"
"""

import argparse
import ast
import json
import operator
import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from baseline_myimpl import build_index, load_chunks, retrieve

load_dotenv()

CHUNKS = load_chunks()
VOCAB, IDF, MATRIX = build_index(CHUNKS)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


# ══════════════════════════════════════════════════════════════
# 1. 工具：@tool 从「类型注解 + docstring」自动生成 schema
#    对照原版：这里不再手写 TOOLS 列表
# ══════════════════════════════════════════════════════════════
@tool
def search_kb(query: str) -> str:
    """查询「单词萌宠」资料库，包括玩法、宠物养成规则、金币获取、账号问题。
    只用于回答游戏相关问题，不能用于数学计算。
    参数 query：要检索的问题，用中文完整短句。"""
    hits = retrieve(query, MATRIX, CHUNKS, VOCAB, IDF, k=3)
    if not hits or hits[0][0] <= 0:
        return "知识库中没有相关内容。"
    return "\n".join(
        f"[片段{i}]（{c['chunk_id']}）{c['text'][:150]}" for i, (_, c) in enumerate(hits, 1)
    )


@tool
def calc(expression: str) -> str:
    """四则运算计算器，支持 + - * / 和括号。凡是需要算数的地方都必须调用它，不要自己心算。
    参数 expression：算式，如 (3*2)*5。"""
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


TOOLS = [search_kb, calc]
# 装饰器生成的 schema 长什么样？取消下一行注释可打印出来看：
# print(json.dumps([t.name for t in TOOLS]), [t.args for t in TOOLS])

# 按名字查表，替代原版的 if name == "search_kb" / elif name == "calc"
TOOL_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = (
    "你是「单词萌宠」的客服助手。可以调用工具来完成任务。\n"
    "规则：\n"
    "1. 问游戏玩法、规则、金币等资料问题时，必须调用 search_kb，不要凭记忆回答。\n"
    "2. 涉及数字计算时，必须调用 calc，不要自己算。\n"
    "3. 工具结果里没有答案，就回答「资料中未提及」，不要猜测。\n"
    "4. 回答不超过 100 字，面向初中生。"
)

MAX_STEPS = 5

# bind_tools 把 schema 传给模型，等价于原版 create() 里的 tools=TOOLS
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    temperature=0,
    max_tokens=400,
).bind_tools(TOOLS)


def run(question: str, verbose: bool = True) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    trace = []

    for step in range(1, MAX_STEPS + 1):
        # LangChain 的 tool_calls 是 dict 列表：{"name":..., "args":..., "id":...}
        # 参数已经被解析成 dict，不需要原版第 171 行的 json.loads
        msg = llm.invoke(messages)

        if not msg.tool_calls:  # 分支一：不调工具，结束
            trace.append(f"第{step}步 结束：直接回答")
            if verbose:
                for line in trace:
                    print("  " + line)
            return msg.content or ""

        messages.append(AIMessage.model_validate(msg))  # 分支二：模型提出调用请求
        for tc in msg.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tool_obj = TOOL_BY_NAME.get(name)
            result = tool_obj.invoke(args) if tool_obj else f"未知工具 {name}"
            trace.append(f"第{step}步 调 {name}({json.dumps(args, ensure_ascii=False)}) → {str(result)[:50]}…")
            # 第三步照样要自己写：结果以 ToolMessage 塞回对话
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    if verbose:
        for line in trace:
            print("  " + line)
    return f"超过 {MAX_STEPS} 步仍未给出答案，已停止（防止死循环）。"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default="宠物饿了怎么办？")
    a = ap.parse_args()
    print(run(a.ask))
