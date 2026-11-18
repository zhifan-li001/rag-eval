# -*- coding: utf-8 -*-
"""
诊断 4：软化系数该不该用？—— 细网格 + 显著性，防止挑一个好参数当结论
=================================================================================
一个待验证的设想：「只要取 Q 和 K 相关性最大的那个就行，不该像注意力那样加权求和。」
这个设想等价于 alpha → ∞ 的硬取最大（MaxSim）。

前期粗网格实验里，alpha=20 在 90 题上给到 86/90，比 MaxSim 的 84 高 2 题。
但粗网格只有 9 个点，86 可能只是那个点的运气。本脚本：
  1. 在 10~60 之间铺细网格，看是一条平台还是一条尖峰；
  2. 对「最优 alpha」与「硬取最大」做逐题配对检验（McNemar 精确检验）；
  3. 分开报 chunk@5 与 chunk@1，因为两者可能反向变化。
"""
import os

# 模型通常已缓存在本地，强制离线避免联网检查卡住。
# 首次使用请先下载模型（见 README「快速开始」第 ② 步），或改用镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import math

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from baseline import build_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
MODEL = "BAAI/bge-small-zh-v1.5"
CACHE = "token_vecs_cache.npz"

chunks = build_chunks()
ids = [c["chunk_id"] for c in chunks]
N = len(ids)
col = {c: i for i, c in enumerate(ids)}

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).eval()


@torch.no_grad()
def token_vecs(texts, batch=8):
    out = []
    drop = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    for s in range(0, len(texts), batch):
        b = texts[s:s + batch]
        enc = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt")
        h = mdl(**enc).last_hidden_state
        for j in range(len(b)):
            keep = [k for k, t in enumerate(enc["input_ids"][j].tolist()) if t not in drop]
            v = h[j][keep]
            out.append(torch.nn.functional.normalize(v, dim=-1) if len(keep) else v)
    return out


def load_90():
    out = []
    for r in csv.DictReader(open("golden_set_100.csv", encoding="utf-8-sig")):
        gs = [g.strip() for g in (r.get("gold_chunk_id") or "").replace(";", "|").split("|") if g.strip()]
        q = (r.get("question") or "").strip()
        if q and gs:
            out.append((r["qid"].strip(), q, gs))
    return out


def load_43():
    o, w = [], []
    for r in csv.DictReader(open("paraphrase_43.csv", encoding="utf-8-sig")):
        gs = [g.strip() for g in r["gold_chunk_id"].split("|") if g.strip()]
        o.append((r["qid"], r["question_orig"], gs))
        w.append((r["qid"], r["question_rewrite"], gs))
    return o, w


TASKS = [("90题·原问法", load_90()), ("43题·原问法", load_43()[0]), ("43题·改写", load_43()[1])]

# --- 向量缓存，避免每次重编码 ---
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    DOC_TV = [torch.from_numpy(a) for a in z["doc"]]
    Q_TV = [torch.from_numpy(a) for a in z["q"]]
    print(f"读缓存 {CACHE}")
else:
    print("编码块 token 向量…")
    DOC_TV = token_vecs([c["text"] for c in chunks])
    print("编码问题 token 向量…")
    Q_TV = token_vecs([it[1] for _, items in TASKS for it in items])
    np.savez(CACHE, doc=np.array([t.numpy() for t in DOC_TV], dtype=object),
             q=np.array([t.numpy() for t in Q_TV], dtype=object))
    print(f"已缓存 {CACHE}")

ALPHAS = [10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 40, 60, 100, 1e4]
sec = lambda t: print("\n" + "=" * 92 + f"\n{t}\n" + "=" * 92)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(p, 1.0)


result = {}
QP = 0
for tname, items in TASKS:
    n = len(items)
    qtv = Q_TV[QP:QP + n]
    QP += n
    gold_of = [{col[g] for g in it[2] if g in col} for it in items]
    M = {a: np.zeros((n, N), dtype=np.float32) for a in ALPHAS}
    for i in range(n):
        q = torch.nn.functional.normalize(qtv[i], dim=-1)
        for j in range(N):
            S = q @ DOC_TV[j].T
            for a in ALPHAS:
                if a >= 1e4:
                    M[a][i, j] = float(S.max(dim=-1).values.mean())
                else:
                    W = torch.softmax(a * S, dim=-1)
                    M[a][i, j] = float((W * S).sum(dim=-1).mean())
    result[tname] = (n, M, gold_of, items)

sec("1 · 细网格：5 题窗口命中数（括号内为 chunk@1）")
head = f"{'alpha':>8}" + "".join(f"{t[:9]:>13}" for t, _ in TASKS)
print(head)
for a in ALPHAS:
    line = f"{a:>8.0f}"
    for tname, _ in TASKS:
        n, M, gold_of, _ = result[tname]
        o = np.argsort(-M[a], axis=1)
        h5 = sum(1 for i in range(n) if gold_of[i] & set(o[i][:5]))
        h1 = sum(1 for i in range(n) if gold_of[i] & set(o[i][:1]))
        line += f"{h5:>6}/{n}({h1:>2})"[:13].rjust(13)
    print(line)

sec("2 · 最优 alpha 与硬取最大（MaxSim）的逐题配对检验")
print(f"{'任务':<14}{'最优α':>8}{'并列α':>22}{'软化':>8}{'MaxSim':>9}{'赢/输':>9}{'p':>9}")
for tname, _ in TASKS:
    n, M, gold_of, _ = result[tname]
    hard = np.argsort(-M[1e4], axis=1)
    h_hard = [bool(gold_of[i] & set(hard[i][:5])) for i in range(n)]
    best = max(ALPHAS, key=lambda a: (sum(1 for i in range(n)
                                        if gold_of[i] & set(np.argsort(-M[a], axis=1)[i][:5])), -a))
    soft = np.argsort(-M[best], axis=1)
    h_soft = [bool(gold_of[i] & set(soft[i][:5])) for i in range(n)]
    win = [i for i in range(n) if h_soft[i] and not h_hard[i]]
    lose = [i for i in range(n) if h_hard[i] and not h_soft[i]]
    plateau = [a for a in ALPHAS if a < 1e4 and
               sum(1 for i in range(n) if gold_of[i] & set(np.argsort(-M[a], axis=1)[i][:5]))
               == sum(1 for i in range(n) if gold_of[i] & set(soft[i][:5]))]
    print(f"{tname:<14}{best:>8.0f}{str([int(x) if x < 1e4 else '∞' for x in plateau]):>22}"
          f"{len(h_soft)-sum(h_hard):>+8}{sum(h_hard):>6}/{n}"
          f"{len(win)}/{len(lose):<6}{mcnemar(len(win), len(lose)):>9.4f}")
    print(f"{'':<14}赢的题 {[result[tname][3][i][0] for i in win]}  输的题 {[result[tname][3][i][0] for i in lose]}")

sec("3 · 稳定性：把 alpha 当作超参，看曲线是一段平台还是一根尖峰")
for tname, _ in TASKS:
    n, M, gold_of, _ = result[tname]
    vals = []
    for a in ALPHAS:
        o = np.argsort(-M[a], axis=1)
        vals.append(sum(1 for i in range(n) if gold_of[i] & set(o[i][:5])))
    soft_part = vals[:-1]
    print(f"  {tname:<14} α∈[10,100] 的命中序列 {soft_part}  → 极差 {max(soft_part)-min(soft_part)}"
          f"，硬取最大 {vals[-1]}")
print("\n  判据：若 α 在 16–32 一段内都 ≥ MaxSim，则「软化有收益」站得住；")
print("        若只有单个 α 冒尖、两侧立刻掉回去，则那 2 题属于噪声，应按硬取最大报。")
