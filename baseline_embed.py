# -*- coding: utf-8 -*-
"""
语义检索：把 108 个 chunk 编码成向量矩阵

做三件事，只做这三件：
  1. 用 baseline.build_chunks() 切出与 baseline.py 完全一样的 108 块（行序必须一致，否则没法对比）
  2. 用本地 bge-small-zh-v1.5 把每块文字压成 512 个数
  3. 存成 vectors_bge.npy + vectors_bge_meta.json，并打印形状自证

不接大模型、不联网、不花钱。这一步的模型是「编码器」，它不会说话，只会把文字变成数。

用法：
    python baseline_embed.py
"""

import os

# 模型通常已缓存在本地，强制离线避免联网检查卡住。
# 首次使用请先下载模型（见 README「快速开始」第 ② 步），或改用镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline import build_chunks

HERE = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VEC_PATH = HERE / "vectors_bge.npy"
META_PATH = HERE / "vectors_bge_meta.json"


def main() -> None:
    # ---------- 1. 切块 ----------
    chunks = build_chunks()
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    print(f"[1/3] 切块完成：{len(chunks)} 块")

    # ---------- 2. 加载模型 ----------
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"[2/3] 模型加载完成：{time.time() - t0:.1f}s")
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") \
        else model.get_sentence_embedding_dimension()
    print(f"      模型输出维度 = {dim}")
    print(f"      模型单次最多吃 = {model.max_seq_length} 个 token")

    # 先量一下最长的块占多少个 token，证明「没有被截断」
    # 注意：token 数 ≠ 字数，中文常常 1 个字 ≈ 1 个 token，但数字/英文会差很多
    tok_lens = [len(x) for x in model.tokenizer(texts, truncation=False)["input_ids"]]
    print(f"      最长块 = {max(tok_lens)} token，最短块 = {min(tok_lens)} token")
    over = sum(1 for n in tok_lens if n > model.max_seq_length)
    print(f"      超过上限被截断的块数 = {over}")

    # ---------- 3. 编码 ----------
    t0 = time.time()
    vecs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  # 先归一化，后面算余弦就是一次点积
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    print(f"[3/3] 编码完成：{time.time() - t0:.2f}s")

    # ---------- 自证 ----------
    print()
    print(f"矩阵形状 = {vecs.shape}")
    print(f"数据类型 = {vecs.dtype}")
    norms = np.linalg.norm(vecs, axis=1)
    print(f"每行长度（应当是 1.0）：min={norms.min():.4f}  max={norms.max():.4f}")
    print(f"第 37 行（下标 36）对应块 = {ids[36]}")
    print(f"该行前 5 个数 = {np.round(vecs[36][:5], 4).tolist()}")

    # ---------- 落盘 ----------
    np.save(VEC_PATH, vecs)
    meta = {
        "model": MODEL_NAME,
        "dim": int(vecs.shape[1]),
        "count": int(vecs.shape[0]),
        "normalized": True,
        "chunk_ids": ids,  # 行序对照表：第 i 行 = chunk_ids[i]
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"已存：{VEC_PATH.name}（{VEC_PATH.stat().st_size / 1024:.0f} KB）")
    print(f"已存：{META_PATH.name}（含 chunk_id 行序对照表）")


if __name__ == "__main__":
    main()
