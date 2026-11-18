# -*- coding: utf-8 -*-
"""
诊断 2：MaxSim 精排 —— 语义捞池子，token 级重排挑块
================================================================
背景（diag_1_ceiling.py 实测）：
  语义 top5 = 78/90，但语义 top30 = 90/90。
  → gold 从来没丢，只是排不进前五。问题在「排序」，不在「召回」。

本脚本对比三种重排器，全部离线、不加新模型：
  A. 语义分原样        —— 对照组
  B. MaxSim           —— late interaction：问题每个 token 去块里找最像的那个，
                          取平均。是 cross-encoder 的廉价近似，不用训练。
  C. RRF(语义, MaxSim) —— 两个名次取倒数相加，k=60，权重事先写死。

防过拟合的写法：
  · 权重和 k 全部事先声明，不在数据上调。
  · 90 题集只用来「报成绩」，43 题改写集用来「验泛化」。
  · 同时报 MaxSim 与语义分的 Spearman 相关 —— 相关太高说明没有增量。
"""
import os

# 模型通常已缓存在本地，强制离线避免联网检查卡住。
# 首次使用请先下载模型（见 README「快速开始」第 ② 步），或改用镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import statistics

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from baseline import Index, build_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
MODEL = "BAAI/bge-small-zh-v1.5"
POOL = 20           # 池子大小（语义 top20 = 89/90 天花板）
RRF_K = 60          # RRF 常数，写死

chunks = build_chunks()
ids = [c["chunk_id"] for c in chunks]
N = len(ids)
col = {c: i for i, c in enumerate(ids)}
idx = Index(chunks)
V = np.load("vectors_bge.npy")
st = SentenceTransformer(MODEL, device="cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).eval()


# ── token 级向量 ────────────────────────────────────────────
@torch.no_grad()
def token_vecs(texts, batch=8):
    """返回 list[Tensor(Li, 512)]，已去 [CLS]/[SEP]/padding，逐 token L2 归一化。"""
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


def maxsim(qv, dv):
    if len(qv) == 0 or len(dv) == 0:
        return 0.0
    return float((qv @ dv.T).max(dim=1).values.mean())


# ── 评估任务 ────────────────────────────────────────────────
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


# 预编码所有块的 token 向量（只需一次）
print(f"块 {N} 个 → 编码 token 向量…")
DOC_TV = token_vecs([c["text"] for c in chunks])

# 预编码所有问题的 token 向量
ALL_Q = []
for _, items in TASKS:
    ALL_Q.extend([it[1] for it in items])
print(f"问题 {len(ALL_Q)} 条 → 编码 token 向量…")
Q_TV = token_vecs(ALL_Q)
QP = 0

sec = lambda t: print("\n" + "=" * 84 + f"\n{t}\n" + "=" * 84)
results = []

for tname, items in TASKS:
    n = len(items)
    qtv = Q_TV[QP:QP + n]
    QP += n
    QE = st.encode([it[1] for it in items], batch_size=16,
                   normalize_embeddings=True, show_progress_bar=False)
    S = QE @ V.T                                     # (n, 108) 语义分

    hitA = hitB = hitC = 0
    rho_pairs, pool_miss = [], 0
    detail = []

    for i, (qid, q, golds, cat) in enumerate(items):
        cand = np.argsort(-S[i])[:POOL]              # 语义捞池子
        goldset = {col[g] for g in golds if g in col}

        # A · 语义分原样
        rankA = list(cand[np.argsort(-S[i][cand])])[:5]

        # B · MaxSim 重排（池子内）
        ms = np.array([maxsim(qtv[i], DOC_TV[j]) for j in cand])
        rankB = list(cand[np.argsort(-ms)])[:5]

        # C · RRF(语义排名, MaxSim 排名)，k=60
        r_sem = np.argsort(np.argsort(-S[i][cand])) + 1
        r_ms = np.argsort(np.argsort(-ms)) + 1
        rrf = 1.0 / (RRF_K + r_sem) + 1.0 / (RRF_K + r_ms)
        rankC = list(cand[np.argsort(-rrf)])[:5]

        okA = bool(goldset & set(int(x) for x in rankA))
        okB = bool(goldset & set(int(x) for x in rankB))
        okC = bool(goldset & set(int(x) for x in rankC))
        hitA += okA; hitB += okB; hitC += okC

        if not (goldset & set(int(j) for j in cand)):
            pool_miss += 1

        # MaxSim 与语义分的 Spearman（池子内）
        if len(cand) > 3:
            a = np.argsort(np.argsort(S[i][cand]))
            b = np.argsort(np.argsort(ms))
            rho_pairs.append(float(np.corrcoef(a, b)[0, 1]))

        detail.append({
            "qid": qid, "category": cat, "question": q,
            "gold": "|".join(golds),
            "sem_top5": "|".join(ids[j] for j in rankA),
            "maxsim_top5": "|".join(ids[j] for j in rankB),
            "rrf_top5": "|".join(ids[j] for j in rankC),
            "hit_sem": int(okA), "hit_maxsim": int(okB), "hit_rrf": int(okC),
            "gold_rank_sem": int(np.where(np.argsort(-S[i]) == list(goldset & set(range(N)))[0])[0][0]) + 1
            if (goldset & set(range(N))) else -1,
        })

    results.append((tname, n, hitA, hitB, hitC, pool_miss, rho_pairs, detail))

    print(f"\n### {tname}   n={n}   池子=语义top{POOL}")
    print(f"  A 语义分原样     {hitA:>3}/{n}  = {hitA/n:.1%}")
    print(f"  B MaxSim 重排    {hitB:>3}/{n}  = {hitB/n:.1%}   ({hitB-hitA:+d})")
    print(f"  C RRF(语义,MaxSim){hitC:>3}/{n}  = {hitC/n:.1%}   ({hitC-hitA:+d})")
    print(f"  池子本身漏掉 gold 的题：{pool_miss}")
    if rho_pairs:
        print(f"  MaxSim vs 语义分 名次相关（池内）：均值 {statistics.mean(rho_pairs):.3f}  "
              f"中位 {statistics.median(rho_pairs):.3f}   ← 越接近 1 说明越没增量")

# ── 逐题变化（90 题集）──────────────────────────────────────
sec("90 题集 · MaxSim 改变结论的题")
tname, n, hA, hB, hC, _, _, det = results[0]
for r in det:
    if r["hit_sem"] != r["hit_maxsim"]:
        d = "赢" if r["hit_maxsim"] > r["hit_sem"] else "输"
        print(f"  {r['qid']} [{r['category']:<8}] MaxSim{d}  {r['question'][:30]}")
        print(f"        gold={r['gold']}")
        print(f"        语义top5：{r['sem_top5']}")
        print(f"        MaxSim ：{r['maxsim_top5']}")

# ── 落盘 ────────────────────────────────────────────────────
for tname, n, hA, hB, hC, _, _, det in results:
    fn = "rerank_" + tname.replace("·", "_") + ".csv"
    with open(fn, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det[0].keys()))
        w.writeheader(); w.writerows(det)
    print(f"已存 {fn}")

print("\n" + "=" * 84)
print("汇总")
print("=" * 84)
print(f"{'任务':<16}{'n':>5}{'语义':>8}{'MaxSim':>9}{'RRF':>9}")
for tname, n, a, b, c, *_ in results:
    print(f"{tname:<16}{n:>5}{a:>6}/{n}{b:>6}/{n}{c:>6}/{n}")
