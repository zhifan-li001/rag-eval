# -*- coding: utf-8 -*-
"""
诊断 3：消融 —— MaxSim 到底是「重排器」还是「检索器」
================================================================
diag_2_maxsim 得到：语义池20 + MaxSim 重排 = 84/90（语义原样 78）。
但有两个问题没答：
  1. 池子 20 是拍出来的，换大小会不会翻？
  2. MaxSim 如果直接从 108 块里排，还要不要语义捞池子？

如果全库 MaxSim 也能到 84，架构就可以砍掉一半：
  不需要 bge 句向量，也不需要两段式，一个 MaxSim 检索器就够。

本脚本一次算全库 MaxSim 矩阵，然后所有组合都从它派生。
"""
import os

# 模型通常已缓存在本地，强制离线避免联网检查卡住。
# 首次使用请先下载模型（见 README「快速开始」第 ② 步），或改用镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from baseline import build_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
MODEL = "BAAI/bge-small-zh-v1.5"

chunks = build_chunks()
ids = [c["chunk_id"] for c in chunks]
N = len(ids)
col = {c: i for i, c in enumerate(ids)}
V = np.load("vectors_bge.npy")
st = SentenceTransformer(MODEL, device="cpu")
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
    rows = list(csv.DictReader(open("golden_set_100.csv", encoding="utf-8-sig")))
    out = []
    for r in rows:
        gs = [g.strip() for g in (r.get("gold_chunk_id") or "").replace(";", "|").split("|") if g.strip()]
        q = (r.get("question") or "").strip()
        if q and gs:
            out.append((r["qid"].strip(), q, gs, (r.get("category") or "").strip()))
    return out


def load_43():
    rows = list(csv.DictReader(open("paraphrase_43.csv", encoding="utf-8-sig")))
    o, w = [], []
    for r in rows:
        gs = [g.strip() for g in r["gold_chunk_id"].split("|") if g.strip()]
        o.append((r["qid"], r["question_orig"], gs, r["category"]))
        w.append((r["qid"], r["question_rewrite"], gs, r["category"]))
    return o, w


TASKS = [("90题·原问法", load_90()),
         ("43题·原问法", load_43()[0]),
         ("43题·改写", load_43()[1])]

print("编码块 token 向量…")
DOC_TV = token_vecs([c["text"] for c in chunks])
allq = [it[1] for _, items in TASKS for it in items]
print(f"编码问题 token 向量（{len(allq)} 条）…")
Q_TV = token_vecs(allq)


def maxsim_row(qv):
    """对一个问题，算它与全部 108 块的 MaxSim 分"""
    out = np.empty(N, dtype=np.float32)
    for j in range(N):
        out[j] = float((qv @ DOC_TV[j].T).max(dim=1).values.mean())
    return out


sec = lambda t: print("\n" + "=" * 84 + f"\n{t}\n" + "=" * 84)

summary = []
QP = 0
for tname, items in TASKS:
    n = len(items)
    qtv = Q_TV[QP:QP + n]
    QP += n
    QE = st.encode([it[1] for it in items], batch_size=16,
                   normalize_embeddings=True, show_progress_bar=False)
    S = QE @ V.T

    t0 = time.perf_counter()
    MS = np.vstack([maxsim_row(qtv[i]) for i in range(n)])
    ms_time = time.perf_counter() - t0

    gold_of = [{col[g] for g in it[2] if g in col} for it in items]
    ok = lambda r: sum(1 for i in range(n) if gold_of[i] & set(r[i][:5]))

    # 1 全库 MaxSim 直接排
    full_ms = [list(np.argsort(-MS[i])) for i in range(n)]
    # 2 全库语义直接排
    full_sem = [list(np.argsort(-S[i])) for i in range(n)]
    # 3 全库 RRF
    rrf_full = []
    for i in range(n):
        a = np.argsort(np.argsort(-S[i])) + 1
        b = np.argsort(np.argsort(-MS[i])) + 1
        rrf_full.append(list(np.argsort(-(1.0 / (60 + a) + 1.0 / (60 + b)))))

    print(f"\n### {tname}   n={n}   MaxSim 全库耗时 {ms_time:.1f}s")
    print(f"  语义分 全库直接排        {ok(full_sem):>3}/{n}")
    print(f"  MaxSim 全库直接排        {ok(full_ms):>3}/{n}")
    print(f"  RRF(语义,MaxSim) 全库    {ok(rrf_full):>3}/{n}")

    # 4 池子大小敏感性
    print(f"  {'池子':<10}{'MaxSim重排':>14}{'语义原样':>12}{'池子漏gold':>12}")
    for K in (5, 10, 20, 30, 50):
        rk_ms, rk_sem, miss = [], [], 0
        for i in range(n):
            cand = np.argsort(-S[i])[:K]
            rk_ms.append(list(cand[np.argsort(-MS[i][cand])]))
            rk_sem.append(list(cand))
            if not (gold_of[i] & set(int(x) for x in cand)):
                miss += 1
        print(f"  top{K:<7}{ok(rk_ms):>10}/{n}{ok(rk_sem):>7}/{n}{miss:>12}")

    summary.append((tname, n, ok(full_sem), ok(full_ms), ok(rrf_full)))

sec("汇总")
print(f"{'任务':<16}{'n':>5}{'语义全库':>11}{'MaxSim全库':>13}{'RRF全库':>11}")
for tname, n, a, b, c in summary:
    print(f"{tname:<16}{n:>5}{a:>8}/{n}{b:>10}/{n}{c:>8}/{n}")
