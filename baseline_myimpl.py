# -*- coding: utf-8 -*-
"""
「单词萌宠」文档检索 · 原生实现
================================================================
不依赖 ES / FAISS / LangChain，只用 numpy 手写一条完整的字面检索链路：

    切块 → 建词表 → 算 IDF → 文本向量化 → 矩阵乘排序 → 命中判定

函数内部逻辑与 baseline.py 逐行对应，差别只在两处：
  · 命名统一为英文（交付物/简历里可读）
  · 去掉 .env、LLM 调用、CLI 参数，只留检索核心

本文件只含检索核心，不含批量评估代码（评估链路见 diag_*.py）。

用法：
    python baseline_myimpl.py                              # 冒烟：语料规模 + 单条查询 top-5
    python baseline_myimpl.py --ask "宠物死了怎么办"
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

CORPUS_DIR = HERE / "corpus"
GOLDEN_CSV = HERE / "golden_set_100.csv"
TARGET_SIZE = 200      # 每块目标字数，与 W2 切块保持一致


# ═══════════════════════════════════════════════════════════════
# 1. 切句 / 切字
# ═══════════════════════════════════════════════════════════════
def split_sentences(paragraph: str) -> list[str]:
    """按句末标点切句，标点保留在句内。

    (?<=[。！？]) 是零宽后顾：只在标点「后面」那个位置切，但不会吃掉标点。
    段落超过 TARGET_SIZE 时必须退到句子级再拼，否则会把一句话拦腰切断。
    """
    return [s.strip() for s in re.split(r"(?<=[。！？])", paragraph) if s.strip()]


def bigrams(text: str) -> list[str]:
    """去掉所有空白，取相邻两字组成一个 gram。

    "宠物死了" -> ["宠物", "物死", "死了"]
    中文不做分词也能做字面匹配，代价是词表比词典大。
    """
    s = "".join(text.split())
    return [s[i:i + 2] for i in range(len(s) - 1)]


# ═══════════════════════════════════════════════════════════════
# 2. 建索引
# ═══════════════════════════════════════════════════════════════
def build_vocab(chunks: list[dict]) -> dict[str, int]:
    """收集全库出现过的 bigram，排序后建立 片段 -> 下标 的映射。"""
    grams = {g for c in chunks for g in bigrams(c["text"])}
    return {g: i for i, g in enumerate(sorted(grams))}


def build_idf(chunks: list[dict], vocab: dict[str, int]) -> np.ndarray:
    """idf = ln((N+1)/(df+1)) + 1

    df 按「出现在多少个块」计 —— 同一块里出现 10 次也只算 1 块。
    分母 +1 防除零；对数外 +1 保证 df=N 时权重也不是 0。
    """
    df = np.zeros(len(vocab), dtype=np.float32)
    for c in chunks:
        for g in set(bigrams(c["text"])):
            df[vocab[g]] += 1.0
    return np.log((len(chunks) + 1) / (df + 1)) + 1.0


def embed_tfidf(text: str, vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    """tf × idf，再做 L2 归一化 —— 归一化之后点积就等于余弦相似度。

    不归一化的话，长块的点积天然更大，检索会永远偏向长 chunk，
    跟内容相关性无关。
    """
    v = np.zeros(len(vocab), dtype=np.float32)
    for g in bigrams(text):
        j = vocab.get(g)          # 词表外的字直接跳过，不能 KeyError
        if j is not None:
            v[j] += 1.0
    v = v * idf
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def build_index(chunks: list[dict]):
    """返回 (vocab, idf, matrix)，matrix 形状为 (块数, 词表大小)。"""
    vocab = build_vocab(chunks)
    idf = build_idf(chunks, vocab)
    matrix = np.vstack([embed_tfidf(c["text"], vocab, idf) for c in chunks])
    return vocab, idf, matrix


# ═══════════════════════════════════════════════════════════════
# 3. 检索 / 命中判定
# ═══════════════════════════════════════════════════════════════
def retrieve(question: str, matrix: np.ndarray, chunks: list[dict],
             vocab: dict[str, int], idf: np.ndarray,
             k: int = 5) -> list[tuple[float, dict]]:
    """返回相似度最高的 k 个块，格式 [(分数, 块字典), ...]，按分数降序。

    核心只有一行：matrix @ 向量 —— 一次矩阵乘等于对全库算 N 个余弦。
    argsort 必须取负号，否则拿到的是分数最低的 k 个。
    """
    scores = matrix @ embed_tfidf(question, vocab, idf)
    order = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in order]


def hit_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> int:
    """正确块里只要有一个进了前 k 名，就算命中。"""
    return int(any(g in ranked_ids[:k] for g in gold_ids))


# ═══════════════════════════════════════════════════════════════
# 4. 数据加载
# ═══════════════════════════════════════════════════════════════
def load_chunks() -> list[dict]:
    """复用 W2 的切块逻辑，得到 108 个块。"""
    from split_preview import chunk_one_doc, load_corpus

    chunks: list[dict] = []
    for name, body in load_corpus(CORPUS_DIR):
        chunks.extend(chunk_one_doc(name, body, TARGET_SIZE))
    return chunks


def load_questions(path: Path = GOLDEN_CSV):
    """逐题产出 (qid, question, gold_ids)，跳过没有标注的行。"""
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            golds = [g.strip() for g in row["gold_chunk_id"].split("|") if g.strip()]
            if golds:
                yield row["qid"], row["question"], golds


# ═══════════════════════════════════════════════════════════════
# 5. 冒烟：只跑单条查询，批量评估留给 w8
# ═══════════════════════════════════════════════════════════════
def _smoke(question: str, k: int = 5) -> None:
    chunks = load_chunks()
    vocab, idf, matrix = build_index(chunks)

    print(f"块数 {len(chunks)}   词表 {len(vocab)}   矩阵 {matrix.shape}")
    print(f"块向量自余弦 = {float(matrix[0] @ matrix[0]):.4f}")
    print("-" * 56)
    print(f"问题：{question}")
    for i, (score, chunk) in enumerate(retrieve(question, matrix, chunks, vocab, idf, k), 1):
        print(f"  [{i}] {score:.3f}  {chunk['chunk_id']}")


if __name__ == "__main__":
    q = "宠物死了怎么办"
    if "--ask" in sys.argv:
        q = sys.argv[sys.argv.index("--ask") + 1]
    _smoke(q)
