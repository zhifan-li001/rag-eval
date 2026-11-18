# -*- coding: utf-8 -*-
"""
诊断 1：量出「召回池」的天花板
================================================================
在动手做重排之前，必须先回答一个问题：
    gold 块到底能不能被捞进池子里？

如果 gold 连前 30 都进不去，重排器再强也救不回来。
所以这张表决定精排能达到的上限。

口径（和前面完全一致，只是把 K 拧大）：
  题源 = golden_set_100.csv 里 90 道可答题（另 10 道应拒答，不进分母）
  gold = 题目标注的标准块，多 gold 时取最容易命中那个
  名次 = gold 在全部 108 个块里的排序位置，1 = 第一
"""
import os

# 模型通常已缓存在本地，强制离线避免联网检查卡住。
# 首次使用请先下载模型（见 README「快速开始」第 ② 步），或改用镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline import Index, build_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

chunks = build_chunks()
ids = [c["chunk_id"] for c in chunks]
N = len(ids)
col = {c: i for i, c in enumerate(ids)}

idx = Index(chunks)
V = np.load("vectors_bge.npy")
st = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

raw = list(csv.DictReader(open("golden_set_100.csv", encoding="utf-8-sig")))
items = []
for r in raw:
    gs = [g.strip() for g in (r.get("gold_chunk_id") or "").replace(";", "|").split("|") if g.strip()]
    q = (r.get("question") or "").strip()
    if not q or not gs:
        continue
    items.append({"qid": r["qid"].strip(), "q": q, "gold": gs,
                  "cat": (r.get("category") or "").strip()})
nq = len(items)
print(f"可答题 {nq} 道   语料块 {N} 个")

Q = st.encode([it["q"] for it in items], batch_size=16,
              normalize_embeddings=True, show_progress_bar=False)
S_sem = Q @ V.T
S_big = np.vstack([idx.matrix @ idx.embed(it["q"]) for it in items])


def rank_of(scores, item):
    order = np.argsort(-scores)
    pos = {int(j): k + 1 for k, j in enumerate(order)}
    return min(pos[col[g]] for g in item["gold"] if g in col)


r_sem = np.array([rank_of(S_sem[i], items[i]) for i in range(nq)])
r_big = np.array([rank_of(S_big[i], items[i]) for i in range(nq)])

# ── 1 · K 曲线 ───────────────────────────────────────────────
sec = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
sec("1 · 召回率随 K 变化（K = 池子里装多少块）")
print(f"{'K':>5}{'二字':>12}{'语义':>12}{'并集池':>12}")
for K in (1, 3, 5, 10, 20, 30, 50, 108):
    b = int((r_big <= K).sum())
    s = int((r_sem <= K).sum())
    u = 0
    for i in range(nq):
        ob = set(np.argsort(-S_big[i])[:K])
        os_ = set(np.argsort(-S_sem[i])[:K])
        if any(col[g] in (ob | os_) for g in items[i]["gold"] if g in col):
            u += 1
    print(f"{K:>5}{b:>7}/{nq}{s:>7}/{nq}{u:>7}/{nq}")

# ── 2 · 名次分布 ─────────────────────────────────────────────
sec("2 · gold 名次分布（108 块里排第几）")
print(f"{'方法':<8}{'中位':>8}{'均值':>8}{'最好':>8}{'最差':>8}{'倒数20':>10}")
for name, r in (("二字", r_big), ("语义", r_sem)):
    print(f"{name:<8}{np.median(r):>8.1f}{r.mean():>8.1f}{r.min():>8d}{r.max():>8d}"
          f"{int((r > N - 20).sum()):>10}")

print(f"\n语义名次更靠前 {int((r_sem < r_big).sum())} 题 / "
      f"二字更靠前 {int((r_big < r_sem).sum())} 题 / 持平 {int((r_sem == r_big).sum())} 题")
print(f"语义排在 50 名之后的题：{int((r_sem > 50).sum())} 题"
      f"（二字 {int((r_big > 50).sum())} 题）")

# ── 3 · 天花板意味着什么 ─────────────────────────────────────
sec("3 · 重排能拿到的最好成绩 = 池子命中数（不是 90）")
for K in (10, 20, 30, 50):
    b = int((r_big <= K).sum())
    s = int((r_sem <= K).sum())
    print(f"  池子只装语义 top{K:<3} → 重排最好 {s}/{nq}"
          f"    装二字 top{K:<3} → {b}/{nq}"
          f"    两池合并 → 见上表")

# ── 4 · 语义池漏掉的题 ──────────────────────────────────────
sec("4 · 语义 top30 都捞不到的题（重排无法挽救的部分）")
bad = [(items[i], int(r_sem[i]), int(r_big[i])) for i in range(nq) if r_sem[i] > 30]
print(f"共 {len(bad)} 题")
for it, rs, rb in bad:
    print(f"  {it['qid']} [{it['cat']:<8}] 语义#{rs:<4} 二字#{rb:<4} {it['q'][:34]}")
    print(f"        gold = {it['gold']}")

# ── 5 · 存明细 ──────────────────────────────────────────────
out = os.path.join(HERE, "diag_rank_detail.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "category", "question", "gold_chunk_id",
                "rank_bigram", "rank_semantic", "gold_in_bigram_top20", "gold_in_sem_top20"])
    for i, it in enumerate(items):
        w.writerow([it["qid"], it["cat"], it["q"],
                    "|".join(it["gold"]), int(r_big[i]), int(r_sem[i]),
                    int(r_big[i] <= 20), int(r_sem[i] <= 20)])
print(f"\n明细已存 {out}")
