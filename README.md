# RAG 检索评测与工具调用 Agent

围绕微信小游戏「单词萌宠」的文档问答场景，完成两个相互衔接的项目：

- **RAG 检索与评测**：11 篇业务文档切成 108 个知识块，配 100 道人工基准题，在同一套判卷逻辑下对比二字 TF-IDF、BGE 语义检索与 MaxSim 精排。
- **工具调用 Agent**：把 RAG 检索和四则计算器封装成工具，由模型自主选择是否调用、调用哪个工具，并根据工具结果继续推理；提供原生 function calling 与 LangChain `@tool` 两种实现。

**RAG 评测结论：二字 TF-IDF 79/90，BGE 语义 78/90，MaxSim 精排 84/90。**

## 工具调用 Agent

Agent 不是固定执行「检索一次 → 生成一次」，而是运行一个由模型驱动的循环：

```text
用户问题 → 模型选择工具与参数 → 执行工具 → ToolMessage 回灌结果
             ↑                                      ↓
             └──────── 继续推理，直到无需调用工具 ────────┘
```

- **两个工具**：`search_kb` 复用 RAG 检索；`calc` 用 AST 白名单计算四则运算，不直接执行 `eval`。
- **工程约束**：显式工具 schema、JSON 参数解析、未知工具处理、最多 5 步，避免无限调用。
- **两种实现**：`agent_min.py` 手写完整 function calling 循环；`agent_tool_langchain.py` 用 `@tool` 自动生成 schema、按名称分发调用，并手动回灌 `ToolMessage`。
- **预设验证场景**：资料问题调用检索，计算问题调用计算器，组合问题可进行多步调用，无关问题不强行使用工具。

| 实现 | 重点 | 文件 |
|---|---|---|
| 原生 OpenAI function calling | 手写 schema、工具分发与消息循环，看清底层机制 | `agent_min.py` |
| LangChain `@tool` | 对照框架封装：自动 schema、`.invoke(args)`、`bind_tools` | `agent_tool_langchain.py` |

## RAG 评测结果

90 题取自 `golden_set_100.csv` 的可答题（另 10 道是应拒答题，用于检查「不知道时会不会瞎编」，不计入命中率）。43 题是另一套独立题集，专门检验问法偏离原文时的鲁棒性。

| 方案 | 90 题·原问法 | 43 题·原问法 | 43 题·改写 |
|---|---|---|---|
| 二字 TF-IDF | 79/90 | **43/43** | 17/43 |
| BGE 句向量（池化） | 78/90 | 38/43 | 20/43 |
| MaxSim（全库排序） | **84/90** | **43/43** | **31/43** |

三列读下来有两件事：

1. **原问法下字面方法几乎满分**（43/43），因为问题就是照着文档措辞写的。这是测试集偏袒字面的直接证据——只看第一列会得出「不需要语义模型」的错误结论。
2. **换个问法，二字掉 26 题、语义掉 18 题、MaxSim 只掉 12 题**。真实用户不会照着文档提问，所以第三列才是这套系统在线上要面对的分布。

`diag_2_maxsim.py` 与 `diag_3_ablation.py` 给出两个口径：池内 top-20 重排为 84/90、29/43；全库排序为 84/90、31/43。差的两题是「语义 top20 没捞到、但 MaxSim 能排进前五」的块。90 题上两口径相同，说明在这份语料上池子只省算力、不改结果。

## 语料与基准

- **语料**：11 篇 Markdown 业务文档（玩法循环、金币经济、复习算法、审核规则、域名限制等），约 1.9 万字。
- **切块**：段落优先；单段超 320 字退到句子级（`。！？；`）切分；贪心合并到约 200 字；不足 60 字的碎片并入前块；孤立标题必须向前绑定正文。实测 **108 块，70–303 字，中位 175 字**，`split_preview.py` 的结尾会打印超长块与过短块计数，两者均为 0。
- **基准**：100 道题由人工编写并标注 gold chunk，覆盖规则、数值、流程、跨块四类。其中 11 道题的答案本就分散在多个不连续块里。

判卷用 LLM 逐题判定「检索结果是否足以回答问题」，同一套 prompt 跑三个方案，避免人工口径漂移。

## 三个关键发现

**1. 瓶颈不在召回，在排序。**

把 BGE 语义检索的 K 从 5 放大到 30，90 题变成 **90/90 全中**——gold 块一直在候选池里，只是没排进前 5。所以后续投入放在精排而不是换更大的 embedding 模型。这一条直接决定了项目后半程的方向。

**2. 重叠切片是负优化，而且单调。**

按常理重叠切片能缓解「答案被切断」，实测相反：0 重叠 80.8%，加到 120 字重叠降到 61.5%，五档单调下降。原因有两层——重叠让块边界变模糊，同一段文字在多块重复出现，挤占 top-5 名额；更根本的是，11 道跨块题里只有 2 道是相邻块切分造成的，另外 9 道**知识本身就分散在文档不同章节**，靠切块参数解决不了。

**3. 看起来有效的调参，过不了显著性检验。**

MaxSim 原始的相似度是逐 token 取最大值后平均。引入温度系数 α 做 softmax 软化（α→∞ 等价于硬取最大）：α=20 时 90 题从 84 升到 86，但 chunk@1 从 60 掉到 59——**最好的那块反而排不到第一**。净胜仅 2 题（2 胜 0 负），McNemar 精确检验 p=0.5，不显著。所以对外只报零超参的 84，86 只作附注。类似地，粗排池取 10/20/30/50 与直接全库检索，90 题结果恒为 84：池子只省算力，不改结果。

## 方法

```
文档 → 切块(108) → 建索引 ─┬─ 二字 TF-IDF 稀疏向量
                          └─ BGE 稠密向量（512 维，221 KB）
                                    ↓
                          top-5 召回 → MaxSim 逐 token 精排 → LLM 生成
```

- **MaxSim**：不做池化。query 的每个 token 向量与块内每个 token 向量求余弦，取每行最大值后平均（late interaction）。273 token 的块不再被压成 1 个向量，相似度不再被平均稀释。
- **服务化**：`POST /ask`，Pydantic 校验（空问题、topk 越界返回 422），检索为空返回 200 + "资料中未提及"而非报错，模型侧超时 504、上游错误 502，索引在模块级只建一次。
- **最小 Agent**：检索与计算器封装成两个工具，模型自主选择调用、记录调用链，设 5 步上限防死循环。实测计算题会调 `calc`、文档题会调 `search_kb`、无关问题（"今天北京天气"）一步都不调。另附 `agent_tool_langchain.py`：用 `@tool` 装饰器重写同一循环，对照展示框架如何自动生成 schema、按名分发工具调用，以及结果仍需手动以 ToolMessage 回灌。

## 快速开始

```bash
pip install -r requirements.txt
```

**① 零依赖（只用标准库）—— 先看切块**

```bash
python split_preview.py               # 逐块打印长度与标题绑定，结尾附切块质量自检
```

**② 需要 BGE 模型（约 400MB，仅首次）**

向量文件是二进制产物，不入库，首次需先生成：

```bash
python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-zh-v1.5')"
python baseline_embed.py              # 编码 108 个块 → vectors_bge.npy（约 220 KB）
```

网络受限时改用镜像（Windows CMD 用 `set`，macOS/Linux 用 `export`）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

向量就位后：

```bash
python diag_1_ceiling.py              # K=5/10/20/30 各档召回上限
python diag_2_maxsim.py               # MaxSim 池内重排，三种重排器横向对比
python diag_3_ablation.py             # 消融：池大小 / 全库 / RRF
python diag_4_alpha.py                # 软化系数细网格 + McNemar 检验（首次编码 token 向量，约 3–5 分钟）
```

**③ 需要自己的 API Key（生成与判卷环节）**

```bash
cp .env.example .env                  # 填入 OPENAI_API_KEY / BASE_URL / MODEL
python baseline.py                    # 二字 TF-IDF 全流程：检索 → 生成 → 判卷
python judge_100.py                   # 对已冻结的检索结果重新判卷
python -m uvicorn api_demo:app --port 8010    # 打开 http://127.0.0.1:8010/docs
python agent_min.py                   # 最小 Agent，观察工具调用链
python agent_tool_langchain.py        # 同一循环的 @tool / LangChain 版（需 langchain-core、langchain-openai）
```

`.env` 已在 `.gitignore` 中，不会被提交。

## 文件说明

| 文件 | 作用 |
|---|---|
| `rag_min.py` | 最小可跑 RAG：切块 → 检索 → 拼 prompt → 生成，约 130 行 |
| `baseline.py` | 二字 TF-IDF 主实现，含 `build_chunks()` 切块与 `Index` 倒排索引 |
| `baseline_embed.py` | BGE 句向量版，池化后单向量检索 |
| `split_preview.py` | 切块规则的可视化验收与质量自检 |
| `diag_1_ceiling.py` | 召回上限诊断：K=5/10/20/30 下的天花板 |
| `diag_2_maxsim.py` | MaxSim 池内重排，三种重排器横向对比 |
| `diag_3_ablation.py` | 消融：池大小敏感性、全库 MaxSim 与 RRF 融合 |
| `diag_4_alpha.py` | 软化系数细网格 + McNemar 配对检验 |
| `judge_100.py` | 100 题判卷，输出逐题命中与失败归因 |
| `api_demo.py` | FastAPI 封装，含参数校验与错误码分层 |
| `agent_min.py` | 最小 Agent：工具调用循环 + 步数上限 |
| `agent_tool_langchain.py` | 同一循环用 `@tool` 装饰器改写：schema 自动生成、工具按名分发、ToolMessage 回灌 |
| `golden_set_100.csv` | 100 题基准（qid / category / question / gold_answer / gold_chunk_id） |
| `paraphrase_43.csv` | 43 道改写题（同一问题的口语化重写） |
| `eval_result_golden_set_100_top5.csv` | 冻结的检索记录与生成回答，判卷输入 |
| `corpus/` | 11 篇语料，即检索对象 |

## 已知局限

- 语料规模小（108 块），所有结论在 100 题量级上成立；`p=0.5` 这类不显著结果本身就是样本量不足的表现。
- 分数阈值 0.05 未经标定：「北京今天天气」这类域外问题的最高分 0.0578 高于阈值，实际靠 prompt 里的拒答约束兜住，不是检索层挡住的。
- 判卷用 LLM 做裁判，存在与生成模型同源偏差的可能，未做人工双盲复核。
- 精排只覆盖 BGE 的前 5 候选（另一配置为全库排序），未与 BM25 等其他稀疏召回做融合。

## 说明

本项目为个人学习项目，代码在 AI 辅助下完成；方案设计、实验设计、指标口径与结论判断由本人负责。评测数据与失败案例均如实保留，包括上面那几条不显著、不理想的结果。
