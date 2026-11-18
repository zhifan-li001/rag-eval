# -*- coding: utf-8 -*-
"""
基线 RAG：二字 TF-IDF 检索 + 生成 + 出处
================================================================
链路：读目录 → 切分 → 建索引 → 检索 → 生成 → 带出处。

与最小版本 rag_min.py 的区别有三条，都是关键：
  1. 语料从「内存里 3 段」换成「corpus/ 目录下 11 篇真实文档」（108 个 chunk）
  2. 每个 chunk 有唯一 chunk_id，检索结果和生成结果都把它带出来
  3. 可以用 golden set 批量跑，算出 Recall@K

第 2 条是整套东西能不能调优的前提：
  检索 top-5 里没有 gold_chunk_id  →  检索错了，换生成模型纯属浪费
  检索 top-5 里有 gold_chunk_id    →  检索对了，答错就是生成的问题
没有 chunk_id，你永远分不清这两种错。

依赖：openai + numpy（不装 LangChain）
用法：
    python baseline.py --ask "宠物死了怎么办"
    python baseline.py --ask "宠物死了怎么办" --topk 5
    python baseline.py --eval golden_set.csv              # 跑评测，输出 eval_result.csv
    python baseline.py --eval golden_set.csv --no-llm     # 只算检索召回，不调模型
    python baseline.py --stats                            # 只看索引规模
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 切分逻辑直接复用预览工具，保证「你看到的切分」和「索引里的切分」是同一份
from split_preview import chunk_one_doc, clean, load_corpus  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
except ImportError:
    pass

CORPUS_DIR = HERE / "corpus"
TARGET_SIZE = 200
DEFAULT_TOPK = 5


# ══════════════════════════════════════════════════════════════
# 1. 建索引：读文档 → 切分 → 打 chunk_id
# ══════════════════════════════════════════════════════════════
def build_chunks() -> list[dict]:
    docs = load_corpus(CORPUS_DIR)
    if not docs:
        sys.exit(f"corpus/ 里没有可用的 .md 文件：{CORPUS_DIR}")
    chunks: list[dict] = []
    for name, body in docs:
        chunks.extend(chunk_one_doc(name, body, TARGET_SIZE))
    return chunks


# ══════════════════════════════════════════════════════════════
# 2. 向量化：字符 bigram + TF-IDF
# ══════════════════════════════════════════════════════════════
# 纯词频的问题在于，问题在于「的」「了」「一」这类高频字组成的大 gram
# 在每个 chunk 里都出现，权重却和有信息量的词一样大。
# 加 IDF 就是给「到处都有的 gram」降权 —— 一行 log 的事，效果差别很明显。

def bigrams(s: str) -> list[str]:
    s = "".join(s.split())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # 词表：所有 chunk 出现过的 bigram
        grams = {g for c in chunks for g in bigrams(c["text"])}
        self.vocab = {g: i for i, g in enumerate(sorted(grams))}

        # 文档频率 df：每个 gram 出现在多少个 chunk 里
        df = np.zeros(len(self.vocab), dtype=np.float32)
        for c in chunks:
            for g in set(bigrams(c["text"])):
                df[self.vocab[g]] += 1.0

        # idf = ln((N+1)/(df+1)) + 1 —— 加 1 防止除零，加外层 1 保证 df=N 时仍有权重
        n = len(chunks)
        self.idf = np.log((n + 1) / (df + 1)) + 1.0

        self.matrix = np.vstack([self.embed(c["text"]) for c in chunks])

    def embed(self, text: str) -> np.ndarray:
        """tf × idf，然后 L2 归一化 —— 归一化之后点积就等于余弦相似度。"""
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for g in bigrams(text):
            j = self.vocab.get(g)
            if j is not None:
                v[j] += 1.0
        v *= self.idf
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def retrieve(self, question: str, k: int) -> list[tuple[float, dict]]:
        """向量计算只有这一行：一次矩阵乘 = 对全库算 N 个余弦。"""
        scores = self.matrix @ self.embed(question)
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]


# ══════════════════════════════════════════════════════════════
# 3. 生成：把 top-k 拼进 prompt，要求带出处
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "你是「单词萌宠」的客服助手。只能依据给定资料回答，不许编造资料里没有的内容。\n"
    "资料里没有答案时，直接回答「资料中未提及」，不要猜测。\n"
    "回答用中文，控制在 150 字以内，尽量口语化，面向初中生。\n"
    "回答末尾用括号注明引用的片段编号，例如（依据片段2）。"
)


def make_client():
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or not key.isascii() or len(key) < 20:
        sys.exit(
            "OPENAI_API_KEY 不可用。请检查项目根目录的 .env：\n"
            "  · 是否还是模板占位符\n"
            "  · 有没有混进中文或全角空格（会报 UnicodeEncodeError）"
        )
    from openai import OpenAI

    return OpenAI(
        api_key=key,
        base_url=os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
        timeout=float(os.getenv("TIMEOUT", "60")),
    )


def generate(client, question: str, hits: list[tuple[float, dict]]) -> tuple[str, int, int]:
    context = "\n".join(
        f"[片段{i}]（{c['chunk_id']}）{c['text']}" for i, (_, c) in enumerate(hits, 1)
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
        temperature=0,
        max_tokens=400,
    )
    return (
        resp.choices[0].message.content or "",
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


# ══════════════════════════════════════════════════════════════
# 4. 三种运行模式
# ══════════════════════════════════════════════════════════════
def safe_generate(client, question: str, hits):
    """把常见异常翻成人话。裸 traceback 对排查没帮助，只会掩盖真正原因。"""
    try:
        return generate(client, question, hits)
    except Exception as e:
        name = type(e).__name__
        msg = str(e)
        if "Authentication" in name or "401" in msg:
            sys.exit(
                "\n[401] API Key 无效或已被吊销。\n"
                "  处理：打开项目根目录的 .env，把 OPENAI_API_KEY 换成新生成的 Key。\n"
                "  注意：Key 只能有 ASCII 字符，不要把中文注释或全角空格粘进去。\n"
                f"  服务端原文：{msg[:120]}"
            )
        if "RateLimit" in name or "429" in msg:
            sys.exit(f"\n[429] 触发限流，等 30 秒再跑。原文：{msg[:120]}")
        if "InsufficientBalance" in msg or "402" in msg:
            sys.exit(f"\n[402] 账户余额不足，去 platform.deepseek.com 充值。原文：{msg[:120]}")
        if "Connection" in name or "Timeout" in name:
            sys.exit(f"\n[网络] 连不上 {os.getenv('OPENAI_BASE_URL')}。"
                     f"检查网络/代理，或把 .env 里的 TIMEOUT 调大。原文：{msg[:120]}")
        raise


def show_hits(question: str, hits: list[tuple[float, dict]]) -> None:
    print("=" * 70)
    print(f"问题：{question}")
    print("-" * 70)
    for rank, (score, c) in enumerate(hits, 1):
        head = c["text"].replace("\n", " ")[:64]
        print(f"  [{rank}] {score:.3f}  {c['chunk_id']:28s} {head}…")


def cmd_ask(question: str, topk: int, use_llm: bool) -> None:
    chunks = build_chunks()
    idx = Index(chunks)
    hits = idx.retrieve(question, topk)
    show_hits(question, hits)

    if not use_llm:
        print("\n[--no-llm] 只做检索，不调模型。")
        return

    client = make_client()
    text, pt, ct = safe_generate(client, question, hits)
    print("\n" + "=" * 70)
    print("回答：", text.strip())
    print("-" * 70)
    print(f"prompt_tokens={pt}  completion_tokens={ct}  "
          f"索引规模={len(chunks)} chunk")


def cmd_stats() -> None:
    chunks = build_chunks()
    lengths = sorted(len(c["text"]) for c in chunks)
    print("=" * 70)
    print(f"语料目录：{CORPUS_DIR}")
    print(f"chunk 总数：{len(chunks)}")
    print(f"平均长度：{sum(lengths) // len(lengths)} 字")
    print(f"最短/中位/最长：{lengths[0]} / {lengths[len(lengths) // 2]} / {lengths[-1]}")
    idx = Index(chunks)
    print(f"词表大小（char bigram）：{len(idx.vocab)}")
    print(f"向量库矩阵形状：{idx.matrix.shape}")
    print("-" * 70)
    print("chunk_id 前 5 个示例：")
    for c in chunks[:5]:
        print(f"  {c['chunk_id']:28s} {c['text'][:40]}…")


def doc_of(chunk_id: str) -> str:
    """从 chunk_id 取文档名：'03_复习算法与间隔#1' -> '03_复习算法与间隔'"""
    return chunk_id.split("#")[0]


def cmd_eval(csv_path: Path, topk: int, use_llm: bool) -> None:
    if not csv_path.exists():
        sys.exit(f"找不到 golden set：{csv_path}")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    if not rows:
        sys.exit("golden set 是空的。")

    chunks = build_chunks()
    idx = Index(chunks)
    client = make_client() if use_llm else None
    by_id = {c["chunk_id"]: c for c in chunks}

    out_rows = []
    # category -> [总题数, 严格命中, 宽松命中]，只统计「有标准块」的题
    stat: dict[str, list[int]] = {}

    for r in rows:
        qid = (r.get("qid") or "").strip()
        q = (r.get("question") or "").strip()
        gold = (r.get("gold_chunk_id") or "").strip()
        cat = (r.get("category") or "未分类").strip() or "未分类"
        if not q:
            continue
        gold_ids = [g.strip() for g in gold.replace(";", "|").split("|") if g.strip()]
        is_refuse = not gold_ids          # 没有标准块 = 这道题的正确行为是拒答

        hits = idx.retrieve(q, topk)
        got_ids = [c["chunk_id"] for _, c in hits]

        if is_refuse:
            # 拒答题不进召回率分母：它不存在「该翻到哪一块」，
            # 要评的是「模型有没有老实说不知道」，只能靠人工判卷。
            recalled = same_doc = None
        else:
            recalled = bool(set(gold_ids) & set(got_ids))
            # 宽松判定：gold 所在的那篇文档被命中任意一块就算「方向对了」。
            # 用途是区分两种失败：
            #   严格失败 + 宽松成功  -> 答案在，只是被切分切到了隔壁块（改切分/加重叠）
            #   严格失败 + 宽松失败  -> 检索方向就错了（换 embedding / 改问题改写）
            same_doc = bool({doc_of(g) for g in gold_ids} & {doc_of(g) for g in got_ids})
            s = stat.setdefault(cat, [0, 0, 0])
            s[0] += 1
            s[1] += int(recalled)
            s[2] += int(same_doc)

        missing = [g for g in gold_ids if g not in by_id]
        if missing:
            print(f"  [警告] {qid} 的 gold_chunk_id 在索引里不存在：{missing}")

        ans = ""
        if client:
            ans, _, _ = safe_generate(client, q, hits)

        out_rows.append({
            "qid": qid,
            "category": cat,
            "question": q,
            "gold_chunk_id": gold,
            "top1": got_ids[0] if got_ids else "",
            f"recall@{topk}": "" if is_refuse else ("1" if recalled else "0"),
            "hit_same_doc": "" if is_refuse else ("1" if same_doc else "0"),
            "retrieved_ids": "|".join(got_ids),
            "gold_answer": (r.get("gold_answer") or "").strip(),
            "model_answer": ans.replace("\n", " ").strip(),
            "is_correct": "",          # ← 留空，人工判卷
            "note": "应拒答：资料里没有答案" if is_refuse else "",
        })

    n_all = len(out_rows)
    n_refuse = sum(1 for r in out_rows if r["note"].startswith("应拒答"))
    n_ans = sum(v[0] for v in stat.values())
    hit_at = sum(v[1] for v in stat.values())
    hit_doc = sum(v[2] for v in stat.values())

    print("=" * 70)
    print(f"golden set：{n_all} 条（可答 {n_ans} 条 + 应拒答 {n_refuse} 条）    top_k={topk}")
    print(f"严格召回 Recall@{topk}：{hit_at}/{n_ans} = {hit_at / max(n_ans, 1):.1%}   （chunk_id 必须完全对上）")
    print(f"宽松召回 同文档命中：{hit_doc}/{n_ans} = {hit_doc / max(n_ans, 1):.1%}   （文档对了就算）")
    print("-" * 70)
    print("按类型拆开看（拒答题不进召回分母）：")
    print(f"  {'类型':10s} {'题数':>4s} {'严格召回':>12s} {'宽松召回':>12s}")
    for cat in sorted(stat, key=lambda k: -stat[k][0]):
        t, h, d = stat[cat]
        print(f"  {cat:10s} {t:>4d} {h:>5d}/{t:<4d}{h / t:>6.1%} {d:>5d}/{t:<4d}{d / t:>6.1%}")
    print("-" * 70)
    print("严格失败、但同文档命中的题（答案在，被切分切散了）：")
    for r in out_rows:
        if r[f"recall@{topk}"] == "0" and r["hit_same_doc"] == "1":
            print(f"  {r['qid']:6s} {r['question'][:30]:32s} gold={r['gold_chunk_id']}")
    print("-" * 70)
    print("严格和宽松都失败的题（检索方向就错了）：")
    for r in out_rows:
        if r[f"recall@{topk}"] == "0" and r["hit_same_doc"] == "0":
            print(f"  {r['qid']:6s} {r['question'][:30]:32s} gold={r['gold_chunk_id']}")
    print("-" * 70)
    print("下一步：打开结果表，人工填 is_correct 列（1 对 / 0.5 半对 / 0 错）。")
    print("       拒答题的判法是「有没有老实说资料里没有」，编造即 0 分。")

    # 文件名带上 golden 名字、topk、是否含模型回答，避免三种结果互相覆盖：
    #   eval_result_{golden}_top{k}.csv          ← 带模型回答（花钱，珍贵，不可再生）
    #   eval_result_{golden}_top{k}_只检索.csv    ← --no-llm，随时可重跑
    # 曾经出过的事故：用 --no-llm 重跑，把 100 条回答覆盖成空白。
    suffix = "_只检索" if not use_llm else ""
    out = csv_path.parent / f"eval_result_{csv_path.stem}_top{topk}{suffix}.csv"
    if out.exists() and use_llm:
        print(f"[注意] {out.name} 已存在，将被本次带回答的结果覆盖。")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n已写入 {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline RAG: TF-IDF retrieval + generation")
    ap.add_argument("--ask", type=str, default="", help="单个问题")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help=f"检索条数（默认 {DEFAULT_TOPK}）")
    ap.add_argument("--no-llm", action="store_true", help="只检索，不调模型")
    ap.add_argument("--eval", type=str, default="", help="用 golden set 跑评测")
    ap.add_argument("--stats", action="store_true", help="只看索引规模")
    args = ap.parse_args()

    if args.stats:
        cmd_stats()
    elif args.eval:
        p = Path(args.eval)
        if not p.is_absolute():
            p = HERE / p
        cmd_eval(p, args.topk, not args.no_llm)
    elif args.ask:
        cmd_ask(args.ask, args.topk, not args.no_llm)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
