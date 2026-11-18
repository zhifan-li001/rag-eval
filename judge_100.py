# -*- coding: utf-8 -*-
"""100 题离线判卷：不调用模型、不产生 API 费用。

输入：eval_result_golden_set_100_top5.csv（冻结的真实回答与检索记录）
      golden_set_100.csv（产品作者复核后的当前标准答案）
输出：judge_100_result.csv（100 条最终结果）
      ../争议题复核.html（4 条争议项的作者结论，待复核为 0）

模型回答、retrieved_ids 和 recall@5 始终保留旧系统实测值；本脚本只同步标准答案、
写入最终判定，不重跑模型，也不把旧回答改写成正确答案。
"""
from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
INPUT = HERE / "eval_result_golden_set_100_top5.csv"
GOLDEN = HERE / "golden_set_100.csv"
OUTPUT = HERE / "judge_100_result.csv"
REVIEW_HTML = HERE.parent / "争议题复核.html"

# 明确错误：核心事实错误、漏掉问题要求的核心规则，或资料有答案却拒答。
WRONG = {
    "Q001": "只讲主循环，漏掉抽奖、偷菜、段位和成就；沿用产品作者此前的 0.5 判定，在二分类中不计正确。",
    "Q002": "漏掉 8 个复习档位的具体间隔，以及答错退回第 1 档。",
    "Q012": "价格表未进入前 5，系统错误地回答资料未提及。",
    "Q013": "未答出每天最多 5 次。",
    "Q014": "未答出 S 评级必须同时满足用时不到 30% 和全部一次拼对。",
    "Q026": "把“可能因抽奖被拒”答成“会”，且漏掉金币不能充值、因此不构成法定货币投入这一关键条件。",
    "Q031": "未答出土豆一次收获 8 个。",
    "Q049": "未答出考研词书 5199 个单词。",
    "Q073": "未答出自行申请约 30 个工作日。",
    "Q074": "未答出官方登记不收费。",
    "Q091": "只答资料未提及，没有直接回答不支持日语。",
    "Q092": "没有直接回答没有家长模式。",
    "Q093": "只答资料未提及，没有直接回答目前不支持平板。",
    "Q094": "只答资料未提及，没有直接回答没有发音朗读。",
    "Q095": "只讲发布流程，没有直接回答正式上线时间尚未确定。",
    "Q096": "只答资料未提及，没有直接回答没有电脑版。",
    "Q097": "只答资料未提及，没有直接回答不能对战。",
    "Q098": "只答资料未提及，没有直接回答目前没有发音、也不区分口音。",
    "Q099": "只答资料未提及，没有直接回答由个人开发者制作。",
    "Q100": "只答资料未提及，没有直接回答没有夜间模式。",
}

# 产品作者 2026-10-30 对 4 条争议项的最终裁决。
RESOLVED = {
    "Q007": ("wrong", "同设备中途退出可以续背，学习数据不会丢；但换设备不能恢复上一设备中途退出的那一单。系统声称换设备也能恢复，与产品事实不符。"),
    "Q047": ("correct", "最高段位「萌宠料理王」的门槛是 200 星，系统核心答案正确。另已补充：200 星不是总星数上限，200 星以上也可以继续累计。"),
    "Q082": ("wrong", "当前微信体验版最多 15 人，不是 50 人；需要更多人时应使用网页版，而不是发布社区解除上限。"),
    "Q084": ("correct", "产品作者确认这条回答没有问题：本题不要求把全部审核材料逐项列全，主要材料与必填项答对即可。"),
}


def main() -> None:
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig")))
    golden = {r["qid"]: r for r in csv.DictReader(GOLDEN.open(encoding="utf-8-sig"))}
    if len(rows) != 100 or len(golden) != 100:
        raise ValueError(f"输入与标准答案都应各 100 条，实际 {len(rows)} / {len(golden)}")

    judged = []
    for original in rows:
        qid = original["qid"]
        # 同步作者确认后的标准答案；冻结回答与检索证据不改。
        row = {**original,
               "gold_answer": golden[qid]["gold_answer"],
               "gold_chunk_id": golden[qid]["gold_chunk_id"]}
        if qid in RESOLVED:
            verdict, reason = RESOLVED[qid]
        elif qid in WRONG:
            verdict, reason = "wrong", WRONG[qid]
        else:
            verdict, reason = "correct", "核心事实与标准答案一致；措辞或详略不同不扣分。"
        judged.append({**row, "verdict": verdict, "reason": reason, "judge_error": ""})

    fields = list(judged[0].keys())
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(judged)

    resolved_cards = []
    for r in judged:
        if r["qid"] not in RESOLVED:
            continue
        label = "正确" if r["verdict"] == "correct" else "错误"
        color = "#166534" if r["verdict"] == "correct" else "#b91c1c"
        resolved_cards.append(f"""<article>
<div class="qid">{html.escape(r['qid'])} · {html.escape(r.get('category',''))}</div>
<h3>{html.escape(r['question'])}</h3>
<p><b>当前标准答案：</b>{html.escape(r['gold_answer'])}</p>
<p><b>冻结的系统回答：</b>{html.escape(r['model_answer'])}</p>
<p class="why" style="color:{color}"><b>作者复核：{label}。</b>{html.escape(r['reason'])}</p>
</article>""")

    counts = Counter(r["verdict"] for r in judged)
    answerable = [r for r in judged if r.get("category") != "refuse"]
    ac = Counter(r["verdict"] for r in answerable)
    assert counts["review"] == 0
    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>100 题 · 争议项复核结果</title>
<style>
body{{margin:0;background:#fafaf9;color:#292524;font:14px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:860px;margin:auto;padding:44px 24px 90px}}h1{{font-size:28px;line-height:1.3}}.sub{{color:#78716c}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0 34px}}.stat{{background:#fff;border:1px solid #e7e5e4;padding:12px 18px;min-width:120px}}
.stat b{{display:block;font-size:23px}}article{{background:#fff;border:1px solid #e7e5e4;border-left:3px solid #16a34a;padding:18px 22px;margin:14px 0}}
.qid{{font-size:12px;color:#a8a29e}}h3{{font-size:17px;margin:4px 0 12px}}p{{margin:8px 0}}.why{{font-weight:500}}
.note{{background:#f0fdf4;border:1px solid #bbf7d0;padding:14px 18px;margin:18px 0}}
</style></head><body><main><h1>4 道争议题已全部复核</h1>
<p class="sub">产品作者已给出最终结论。本页保留旧系统回答作为证据，不重跑模型、不产生 API 费用。</p>
<div class="stats"><div class="stat"><b>{counts['correct']}</b>最终正确</div><div class="stat"><b>{counts['wrong']}</b>最终错误</div><div class="stat"><b>0</b>待复核</div></div>
<div class="note">90 道可答题中：{ac['correct']} 道正确、{ac['wrong']} 道错误；10 道功能边界题均未达到“先直接回答”的要求。总计 78 正确 / 22 错误。</div>
{''.join(resolved_cards)}
</main></body></html>"""
    REVIEW_HTML.write_text(doc, encoding="utf-8")

    print(f"correct={counts['correct']} wrong={counts['wrong']} review={counts['review']}")
    print(f"90 道可答题：correct={ac['correct']} wrong={ac['wrong']} review={ac['review']}")
    print(f"全部结果：{OUTPUT}")
    print(f"复核结论：{REVIEW_HTML}")


if __name__ == "__main__":
    main()
