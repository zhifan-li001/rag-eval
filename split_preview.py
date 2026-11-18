# -*- coding: utf-8 -*-
"""
切块预览与质量自检（只用标准库）

作用：语料写作和切块调参时，随时可以看「这些字会被切成什么样」，
      避免写完两万字才发现切分全是碎片、或者 chunk 过长导致检索不准。

用法：
    python split_preview.py                  # 只看统计
    python split_preview.py --show 3         # 看前 3 个 chunk 的全文
    python split_preview.py --show 0         # 看全部
    python split_preview.py --export chunks.csv   # 导出 chunk 清单（带 chunk_id）
    python split_preview.py --size 200       # 改目标长度（默认 200 字）

设计原则（这套规则决定了检索效果的上限）：
    1. 先按「空行」分成段落 —— 段落是你写作时的语义单位
    2. 段落太长再按「句末标点」切 —— 保证不会把一个句子切成两半
    3. 然后贪心合并小句子，凑到接近目标长度 —— 太碎的 chunk 检索不准
    4. chunk 不跨文档 —— 否则「来源」就说不清了
"""

import sys
import re
import csv
import argparse
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
CORPUS_DIR = Path(__file__).parent / "corpus"
TARGET_SIZE = 200          # 每个 chunk 的目标字数
MAX_SIZE = 320             # 超过这个就不再合并了
MIN_SIZE = 60              # 小于这个的碎片，尽量并进前一块

# 句末标点：中文的 。！？ 和英文的 .!?  后面可能跟引号
SENT_END = re.compile(r"(?<=[。！？；!?;])")
# HTML 注释（写作规范提示不要算进语料）
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
# markdown 标题行
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")


def clean(raw: str) -> str:
    """去掉注释、压缩多余空行。保留标题行（它是重要的检索线索）。"""
    raw = HTML_COMMENT.sub("", raw)
    lines = []
    for ln in raw.splitlines():
        s = ln.rstrip()
        # 跳过纯分隔线
        if re.fullmatch(r"-{3,}|\*{3,}", s.strip()):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """按空行切段。

    关键处理：孤立的标题行（整段只有一个 # 开头）必须和它下面那段合并。
    否则标题会被贪心合并塞进上一块的末尾，变成「上一段内容 + 下一节标题」，
    检索时这条 chunk 的语义就脏了 —— 这是很常见但很难发现的切分 bug。
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    pending = ""          # 攒着的孤立标题
    for b in blocks:
        lines = b.splitlines()
        is_only_heading = len(lines) == 1 and bool(HEADING.match(lines[0]))
        if is_only_heading:
            pending = (pending + "\n" + b).strip() if pending else b
            continue
        out.append((pending + "\n" + b).strip() if pending else b)
        pending = ""
    if pending:                      # 文档以标题结尾
        out.append(pending)
    return out


def split_sentences(block: str) -> list[str]:
    """把一段切成句子。markdown 的列表项 '-' 开头也算独立单位。"""
    # 列表项各自成句更安全
    if all(ln.lstrip().startswith(("-", "*", "|")) or HEADING.match(ln) for ln in block.splitlines()):
        items = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(items) > 1:
            return items
    parts = [p.strip() for p in SENT_END.split(block) if p.strip()]
    return parts or [block]


def chunk_one_doc(name: str, text: str, target: int) -> list[dict]:
    """把一篇文档切成若干 chunk，返回 [{chunk_id, source, text, nchar}]"""
    chunks = []
    buf = ""

    def flush():
        nonlocal buf
        s = buf.strip()
        if s:
            chunks.append(s)
        buf = ""

    for block in split_paragraphs(text):
        # 段落本身不长，直接当一块（标题+内容保持在一起）
        if len(block) <= MAX_SIZE:
            if len(buf) + len(block) + 1 <= target:
                buf = (buf + "\n" + block).strip() if buf else block
            else:
                flush()
                buf = block
            continue

        # 段落过长，退到句子级
        for sent in split_sentences(block):
            if len(sent) > MAX_SIZE:
                # 单句还是太长（罕见），硬切
                for i in range(0, len(sent), target):
                    piece = sent[i:i + target]
                    if len(buf) + len(piece) <= target:
                        buf = buf + piece
                    else:
                        flush()
                        buf = piece
                continue
            if len(buf) + len(sent) <= target:
                buf = buf + sent
            else:
                flush()
                buf = sent
    flush()

    # 把过碎的块并进前一块
    merged = []
    for c in chunks:
        if merged and len(c) < MIN_SIZE and len(merged[-1]) + len(c) <= MAX_SIZE:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)

    stem = Path(name).stem
    return [
        {
            "chunk_id": f"{stem}#{i + 1}",
            "source": name,
            "text": c,
            "nchar": len(c),
        }
        for i, c in enumerate(merged)
    ]


def load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    if not corpus_dir.exists():
        sys.exit(f"找不到语料目录：{corpus_dir}\n请先建 corpus/ 并放入 .md 文件。")
    files = sorted(corpus_dir.glob("*.md"))
    docs = []
    for f in files:
        if f.name.lower().startswith(("readme", "说明")):
            continue
        raw = f.read_text(encoding="utf-8", errors="ignore")
        body = clean(raw)
        if body:
            docs.append((f.name, body))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=TARGET_SIZE, help="每个 chunk 目标字数")
    ap.add_argument("--show", type=int, default=-1, help="显示前 N 个 chunk 全文，-1 不显示，0 显示全部")
    ap.add_argument("--export", type=str, default="", help="导出 chunk 清单到 csv")
    args = ap.parse_args()

    docs = load_corpus(CORPUS_DIR)
    if not docs:
        sys.exit("corpus/ 里没有可用的 .md 文件。")

    all_chunks = []
    print("=" * 68)
    print(f"语料目录：{CORPUS_DIR}")
    print(f"目标 chunk 长度：{args.size} 字")
    print("=" * 68)
    print(f"{'文档':38s}{'字数':>8s}{'chunk 数':>10s}")
    print("-" * 68)
    for name, body in docs:
        cs = chunk_one_doc(name, body, args.size)
        all_chunks.extend(cs)
        print(f"{name[:36]:38s}{len(body):>8d}{len(cs):>10d}")
    print("-" * 68)
    total_chars = sum(len(b) for _, b in docs)
    n = len(all_chunks)
    lens = sorted(c["nchar"] for c in all_chunks)
    print(f"{'合计':38s}{total_chars:>8d}{n:>10d}")
    print()
    print(f"  chunk 平均长度：{total_chars // max(n, 1)} 字")
    print(f"  最短 / 中位 / 最长：{lens[0]} / {lens[n // 2]} / {lens[-1]} 字")

    # ── 切块质量自检 ──
    print()
    print("─" * 68)
    print("切块质量自检")
    over = sum(1 for x in lens if x > 320)
    under = sum(1 for x in lens if x < 60)
    print(f"  块长区间：{lens[0]}–{lens[-1]} 字（设计目标 70–320）")
    print(f"  超长块（>320 字）：{over} 个" + ("（句子级切分生效）" if over == 0 else "（检查切分阈值）"))
    print(f"  过短块（<60 字）：{under} 个" + ("（碎片已并入前块）" if under == 0 else "（检查合并逻辑）"))
    print("─" * 68)

    if args.show >= 0:
        show_n = n if args.show == 0 else min(args.show, n)
        print(f"\n前 {show_n} 个 chunk 的内容：\n")
        for c in all_chunks[:show_n]:
            print(f"[{c['chunk_id']}]  {c['nchar']} 字  源：{c['source']}")
            body = c["text"] if len(c["text"]) <= 400 else c["text"][:400] + "…"
            for line in body.splitlines():
                print("    " + line)
            print()

    if args.export:
        out = Path(args.export)
        if not out.is_absolute():
            out = Path(__file__).parent / out
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chunk_id", "source", "nchar", "text"])
            for c in all_chunks:
                w.writerow([c["chunk_id"], c["source"], c["nchar"], c["text"].replace("\n", " ")])
        print(f"\n已导出 {n} 个 chunk -> {out}")
        print("拿去当 golden_set 的参照：出题时对着这张表挑 chunk_id。")


if __name__ == "__main__":
    main()
