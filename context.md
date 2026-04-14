# 毕业设计方案：基于 Revisitable Memory 的考研/考公智能问答助手

> 本文档记录毕设整体思路、创新点设计、数据来源与技术路线。
> 参考论文：ReMemR1 - *Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents* (ICLR 2026)

---

## 一、毕设定位与核心思路

### 1.1 借鉴论文的哪个核心机制

ReMemR1 论文的核心贡献有两个：
1. **Callback 机制**：模型在逐块读取长文档时，可以主动发出 `<recall>query</recall>` 指令，检索**之前读过的历史 memory 片段**，实现非线性的记忆回溯。
2. **Multi-Level Reward**：用 RL（GRPO）训练模型学会何时触发 recall、如何更新 memory。

**你的毕设借鉴的是第一点**（Callback/Recall 机制），不需要重新训练模型，通过**提示工程 + 外部记忆管理模块**在应用层复现这一思想，并适配到考研/考公问答场景。

### 1.2 为什么考研/考公是合适的场景

考研/考公问答具有以下特性，天然契合 Revisitable Memory 的设计动机：

| 场景特性 | 与论文机制的对应 |
|---|---|
| 用户多轮对话，前期提到的院校/专业/分数会在后期被重新引用 | Recall 历史对话中的关键事实 |
| 政策文件（招生简章、大纲）篇幅长达数万字，需逐段理解 | 分块读取 + Memory 更新 |
| 用户会改变目标（换学校/换专业），需修正已有记忆 | Memory Update/Revise |
| 多校对比问题需同时参考多条历史记录 | 多条 Memory 的并行 Recall |

### 1.3 毕设题目建议

- **主推**：基于可回溯记忆机制的考研/考公长文档智能问答系统设计与实现
- 备选：融合 Callback 记忆机制的考研备考问答 Agent 研究
- 备选：面向长上下文咨询场景的可修正记忆问答系统

---

## 二、系统整体架构

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────────────┐
│                   问答 Agent 主流程                    │
│                                                      │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ 意图识别 │───▶│  RAG 检索    │───▶│  Prompt 组装 │ │
│  └─────────┘    │ (政策知识库) │    └──────┬──────┘ │
│                 └──────────────┘           │        │
│                                            ▼        │
│  ┌──────────────────────────────────────────────┐   │
│  │              Memory 管理模块（核心创新）        │   │
│  │                                              │   │
│  │  Short-term Memory   Long-term Memory        │   │
│  │  (当前会话摘要)        (用户档案/历史)          │   │
│  │                                              │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │        Recall 模块（借鉴 ReMemR1）   │    │   │
│  │  │  检测当前问题是否需要回溯历史记忆      │    │   │
│  │  │  触发 <recall>query</recall> 检索    │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────┘   │
│                            │                        │
│                            ▼                        │
│                     LLM 生成回答                     │
│                            │                        │
│                            ▼                        │
│                    Memory Update                    │
│                (更新/修正/遗忘旧记忆)                │
└──────────────────────────────────────────────────────┘
```

### 知识库存储策略说明

**本系统采用完整文档存储，而非预先切块存储。**

传统 RAG 在构建知识库时会预先把文档切成小块（如每块 500 token）存入向量库，检索时返回若干个小块直接拼接给 LLM。这种做法有一个明显缺陷：切块会破坏文档内部的上下文连贯性，导致跨段落的信息无法被完整理解。

本系统的策略：

```
构建阶段：一份招生简章 = 知识库里的一个完整文档（不切块）
                ↓
检索阶段：用 query 检索，返回 1~3 份最相关的完整文档
                ↓
阅读阶段：对每份完整文档应用 ReMemR1 的分块+记忆机制
          （在这里才按 chunk_size 切块，chunk by chunk 读）
                ↓
回答阶段：基于积累的 memory 生成最终答案
```

这样做的好处：
- RAG 负责"找到哪份文档"，保证检索精度
- ReMemR1 机制负责"读懂这份长文档"，保证上下文完整
- 两者分工明确，互不干扰

---

## 三、创新点设计（可直接写进论文）

### 创新点 1：面向考研/考公场景的 Prompted Revisitable Memory 机制

**背景**：ReMemR1 通过 RL 训练模型学会在何时 recall、如何 update memory。但训练需要大量 GPU 资源。本毕设在**推理阶段**通过精心设计的 Prompt，让通用 LLM（如 Qwen2.5-7B）复现这一能力。

**实现方式**：

设计如下 Memory Agent Prompt 框架（参考 `recurrent/impls/memory_revisit.py` 中的 TEMPLATE）：

```
你是一个考研/考公问答助手。你将看到：用户当前的问题、当前对话摘要（memory）
和一段检索到的政策文档（section）。

请按以下格式输出：
- 在 <thinking>...</thinking> 中写出你的推理过程
- 如果当前问题依赖之前对话中的信息，输出 <recall>查询内容</recall> 来回溯历史记忆
- 更新记忆：<update>更新后的对话摘要</update>
- 最终输出回答

<memory>{当前记忆摘要}</memory>
<recalled_memory>{回溯到的历史记忆}</recalled_memory>
<section>{RAG检索到的政策文档片段}</section>
<question>{用户问题}</question>
```

**创新点**：将原论文针对单篇文档线性阅读的 recall 机制，**扩展到多轮对话场景**下跨轮次的信息回溯，适应考研咨询中用户反复引用历史信息的需求。

---

### 创新点 2：完整文档存储 + 召回后分块阅读（Full-doc RAG + ReMemR1）

**背景**：传统 RAG 在存储阶段预先切块（如每块 500 token），检索时返回若干碎片拼接给 LLM，跨段落的关键信息极易丢失。

**本文方案**：

```
传统 RAG：  存储[块1][块2][块3]... → 检索返回碎片 → LLM 直接用碎片回答
                                        ↑ 上下文断裂，多跳推理失败

本文方案：  存储[完整文档A][完整文档B]... → 检索返回完整文档
                                        ↓
                                   ReMemR1 分块阅读
                                   chunk by chunk + memory
                                        ↓
                                   基于 memory 回答
```

**与传统 RAG 的 Trade-off 分析**：

| 维度 | 传统 RAG（预切块）| 本文方案（完整文档）|
|---|---|---|
| 覆盖广度 | 高（多文档多片段并行召回）| 低（1~3 份完整文档）|
| 单文档理解深度 | 低（只看到碎片，上下文断裂）| 高（逐块读取，memory 积累全文信息）|
| 多跳推理能力 | 差（跨段落信息丢失）| 好（memory 保留段落间依赖）|
| 适合场景 | 宽泛检索、碎片知识 | 长文档、上下文强依赖 |

本文方案以完整文档为检索粒度，单次查询召回的来源数量少于传统预切块 RAG，但这一取舍在考研/考公场景下是合理的：用户问题与权威政策文档之间存在高度对应关系（如"北大计算机复试线"的答案几乎 100% 在北大招生简章内），且关键信息（复试线、报录比、复试方式）在同一文档内部存在强依赖，必须完整读取才能支持多跳推理。因此，本场景对检索精度的要求高于覆盖广度。

**论文写法**：
> 传统 RAG 系统在索引阶段对文档进行预分块，检索时从多个位置召回碎片拼接给 LLM，虽覆盖广度较高，但跨段落的上下文信息易丢失，难以支持多跳推理。本文提出以完整文档为粒度构建知识库，单次检索召回的文档数量少于传统方法，但对命中文档引入 ReMemR1 的分块记忆读取机制进行逐块理解，显著提升单文档的信息提取深度。考研/考公问答场景中，用户问题与权威政策文档存在高度对应关系，检索精度优先于覆盖广度，本文方案的设计取舍在此场景下具有合理性，相较传统 RAG 在多跳问答任务上具有明显优势。

---

### 创新点 3：结构化分层记忆（Structured Hierarchical Memory）

传统对话系统只保存聊天历史，本系统设计两层记忆：

**短期记忆（Session Memory）**：当前会话摘要，随对话动态更新
```json
{
  "session_id": "xxx",
  "user_profile": {
    "target_school": "中科大",
    "target_major": "计算机科学",
    "score_range": "380-400",
    "exam_type": "学硕",
    "region": "安徽"
  },
  "query_history": ["中科大计算机复试线是多少", "报录比怎么样"],
  "key_facts": ["中科大计算机2024年复试线340分", "报录比约8:1"],
  "last_updated": "2026-04-13T10:00:00"
}
```

**长期记忆（Long-term Memory）**：跨会话的用户偏好和知识更新
```json
{
  "user_id": "xxx",
  "preferences": {"preferred_regions": ["华东"], "avoid": ["数学一"]},
  "knowledge_updates": [
    {"topic": "中科大计算机分数线", "content": "...", "version": "2024", "timestamp": "..."}
  ]
}
```

**核心操作**：
- `write(key, value, timestamp)`：写入新知识/状态
- `recall(query, k=3)`：TF-IDF 或向量检索历史记忆（直接复用 `recurrent/impls/tf_idf_retriever.py`）
- `revise(key, new_value)`：修正过时/错误记忆
- `forget(key, decay_threshold)`：基于时间衰减遗忘低频记忆

---

### 创新点 3：时序感知的知识可信度管理

考研/考公信息有强时效性（分数线每年更新，大纲每年可能修订），设计**带版本号和有效期的知识条目**：

```json
{
  "knowledge_id": "kau_001",
  "topic": "中科大计算机学硕国家线",
  "content": "A区350，B区335",
  "valid_year": 2024,
  "source": "研招网",
  "confidence": 0.95,
  "expires_at": "2025-03-31"
}
```

当检索到多条同主题知识时，按 `valid_year` 降序优先，`confidence` 加权，自动识别过期条目并提示用户确认。

**论文中对应的写法**：
> 针对考研/考公信息强时效性的特点，本文在 Revisitable Memory 的基础上引入时序感知模块，为每条知识条目添加版本号、有效期和置信度权重。系统在 recall 时优先返回最新版本知识，并对即将过期的条目触发自动更新提醒，解决传统 RAG 系统无法区分知识时效性的问题。

---

### 创新点 4：考研/考公中文长文档问答评测基准（可选，加分项）

自行构建一个小规模但高质量的评测集：
- 从官方文件中提取 200~500 条 QA 对
- 设计多跳问题（需要跨文档、跨段落推理）
- 评测指标：Exact Match、F1、Memory Recall Rate、Cross-turn Consistency

---

## 四、技术选型与实现路线

### 4.1 技术栈

| 模块 | 技术选型 | 理由 |
|---|---|---|
| LLM 骨干 | Qwen2.5-7B-Instruct 或调用 API | 中文能力强，支持长上下文 |
| RAG 框架 | LangChain + LlamaIndex | 成熟，文档丰富 |
| 向量数据库 | ChromaDB 或 FAISS | 轻量，易部署 |
| 记忆检索 | TF-IDF（参考 `recurrent/impls/tf_idf_retriever.py`）+ 向量检索混合 | 直接复用论文代码 |
| 短期记忆存储 | Python dict / Redis | 会话级 |
| 长期记忆存储 | SQLite | 轻量持久化 |
| Web 界面 | Gradio 或 Streamlit | 快速搭建 Demo |
| 文档解析 | PyMuPDF（PDF）、BeautifulSoup（HTML） | 处理政策文档 |

### 4.2 Memory 模块代码结构（参考 ReMemR1）

```
memory/
├── memory_store.py        # 记忆存储与 CRUD 操作
├── recall_engine.py       # Recall 检索（复用 tf_idf_retriever.py）
├── memory_updater.py      # 对话后更新/修正记忆
├── temporal_manager.py    # 时序感知，处理知识过期
└── schema.py              # 记忆数据结构定义
```

### 4.3 核心工作流

```python
def answer(user_query, session_memory, long_term_memory, knowledge_base):
    # Step 1: RAG 检索知识库（返回完整文档，不是碎片块）
    full_docs = knowledge_base.search(user_query, k=3)  # 返回 1~3 份完整文档

    # Step 2: 对每份完整文档用 ReMemR1 方式逐块阅读，积累 memory
    doc_memory = "暂无记忆"
    doc_history_memory = set()
    doc_recalled_memory = "未召回任何记忆"

    for full_doc in full_docs:
        # 命中缓存则跳过重复阅读（热门院校文档只需读一次）
        doc_id = full_doc.metadata["doc_id"]
        if doc_id in doc_memory_cache:
            doc_memory = doc_memory_cache[doc_id]
            continue

        chunks = split_into_chunks(full_doc, chunk_size=1000)  # 此处才切块
        for chunk in chunks:
            prompt = build_reading_prompt(
                question=user_query,
                memory=doc_memory,
                recalled_memory=doc_recalled_memory,
                chunk=chunk,
            )
            response = llm.generate(prompt)

            # 解析 <recall> 标签，检索历史 memory 快照
            recall_query = parse_recall_query(response)
            if recall_query:
                doc_recalled_memory = tfidf_retriever.top1_retrieve(
                    recall_query, doc_history_memory
                )

            # 解析 <update> 标签，更新当前 memory
            doc_memory = parse_update_memory(response)
            doc_history_memory.add(doc_memory)

        # 存入缓存，下次同一文档直接复用
        doc_memory_cache[doc_id] = doc_memory

    # Step 3: 判断是否需要 recall 历史对话记忆（跨轮次）
    recall_query = detect_recall_need(user_query, session_memory)
    recalled = recall_engine.retrieve(recall_query, long_term_memory) if recall_query else None

    # Step 4: 组装最终回答 Prompt
    final_prompt = build_final_prompt(
        question=user_query,
        doc_memory=doc_memory,           # 文档阅读积累的记忆
        session_memory=session_memory,   # 当前会话记忆
        recalled_memory=recalled,        # 历史对话召回
    )

    # Step 5: LLM 生成最终回答
    answer = llm.generate(final_prompt)

    # Step 6: 更新会话记忆和长期记忆
    session_memory.update(user_query, answer)
    long_term_memory.write(user_query, answer)

    return answer
```

---

## 五、知识库数据来源（详细）

### 5.1 考研数据

#### 官方数据源

| 数据 | 来源网站 | 获取方式 | 格式 |
|---|---|---|---|
| 研究生招生信息 | https://yz.chsi.com.cn | 网页爬取/手动下载 | HTML → 文本 |
| 各年国家线（总分+单科）| 教育部/研招网公告 | PDF/HTML 下载 | PDF → 文本 |
| 34所自划线院校分数线 | 各校研究生院官网 | 爬取 | HTML |
| 考研数学大纲 | 教育部教育考试院 | 官方 PDF | PDF |
| 考研英语大纲 | 教育部教育考试院 | 官方 PDF | PDF |
| 考研政治大纲 | 高等教育出版社官网 | 官方 PDF | PDF |
| 院校专业目录（招生简章）| 各院校研究生院 | 爬取或手动整理 | PDF/HTML |

#### 开源/公开整理数据集

| 数据集 | 获取地址 | 内容 |
|---|---|---|
| 考研分数线历年整理 | https://github.com/搜索 `kaoyan-score` | CSV 格式历年国家线和院校线 |
| HotpotQA（论文原始数据）| https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa | 可用于对比实验 |
| LongBench | https://huggingface.co/datasets/THUDM/LongBench | 长文本理解评测基准，中英文 |
| DRCD（中文阅读理解）| https://huggingface.co/datasets/clue/drcd | 中文长文档 QA，可做训练数据 |

#### 可爬取的社区数据（注意合规）

- **考研论坛**（kaoyan.com）：院校讨论、经验贴，含大量真实问答
- **知乎考研话题**：公开问答，适合提取 QA 对
- **小木虫学术科研论坛**：硕博讨论，学科专业信息

### 5.2 考公数据

#### 官方数据源

| 数据 | 来源网站 | 获取方式 |
|---|---|---|
| 国考招考公告 | https://www.scs.gov.cn | PDF 下载 |
| 国考职位表 | 国家公务员局 | Excel 下载（每年10月公布）|
| 行政职业能力测验大纲 | 国家公务员局 | PDF 下载 |
| 申论考试大纲 | 国家公务员局 | PDF 下载 |
| 各省省考招录公告 | 各省人力资源和社会保障厅官网 | PDF/HTML |
| 历年国考行测真题 | 公考宝典等公开平台 | 文本格式（需确认版权）|

#### 公开整理数据

| 数据集 | 来源 | 内容 |
|---|---|---|
| 公务员考试 QA 数据 | Hugging Face 搜索 `gongkao`、`civil servant exam` | 行测知识点 QA |
| C3（中文多选阅读理解）| https://huggingface.co/datasets/clue/c3 | 包含政府/社会类长文档 |
| CAIL-2018 行政类 | https://github.com/china-ai-law-challenge | 含行政法规 QA |

### 5.3 建议最小数据集（毕设够用）

作为毕设，建议构建如下最小可用知识库：

```
knowledge_base/
├── graduate_exam/
│   ├── national_score_lines/     # 2020-2024 国家线（手动整理CSV）
│   ├── syllabi/                  # 数学、英语、政治大纲 PDF
│   ├── school_profiles/          # 20-30 所热门院校招生简章（手动下载）
│   └── faq/                      # 手动整理的高频问答100条
├── civil_service/
│   ├── exam_announcements/       # 近3年国考公告
│   ├── position_tables/          # 近3年职位表 Excel
│   ├── syllabi/                  # 行测/申论大纲
│   └── faq/                      # 手动整理高频问答100条
└── evaluation_set/
    ├── single_hop_qa.json        # 单跳QA（直接从文档提取，100条）
    ├── multi_hop_qa.json         # 多跳QA（需跨段落推理，50条）
    └── cross_turn_qa.json        # 跨轮次QA（测试recall能力，50条）
```

---

## 六、对比实验设计

为了量化创新点的效果，设计以下对比实验：

| 系统 | 描述 |
|---|---|
| **Baseline-NoMemory** | 纯 RAG（预切块存储），无记忆模块 |
| **Baseline-FullHistory** | 把全部对话历史塞进 context（标准 ChatBot 做法）|
| **Baseline-Summary** | 对话历史做滚动摘要（常见做法）|
| **本文系统-NoRecall** | 完整文档存储 + Memory，但无 Recall 机制 |
| **本文系统-WithRecall** | 完整系统（完整文档存储 + Memory + Recall + 时序管理）|
| **本文系统-WithCache** | 完整系统 + 文档 Memory 缓存（评测实际部署效率）|

**准确率指标**：

| 指标 | 定义 | 测试集 |
|---|---|---|
| Accuracy | 答案是否正确（Exact Match / F1）| single_hop_qa |
| Cross-turn Consistency | 长对话中前后回答是否一致 | cross_turn_qa |
| Recall Hit Rate | 需要 recall 时是否成功找回相关记忆 | cross_turn_qa |
| Multi-hop Accuracy | 多跳推理问题准确率 | multi_hop_qa |

**速度与效率指标**：

| 指标 | 定义 | 说明 |
|---|---|---|
| Avg Response Time (首次) | 文档未缓存时的平均响应时间 | 体现方法的时间代价 |
| Avg Response Time (缓存) | 文档已缓存时的平均响应时间 | 体现缓存的优化效果 |
| LLM Call Count | 回答一个问题平均调用 LLM 次数 | 反映计算开销 |

**预期 trade-off 结论（可写入论文）**：

> 本文方法在多跳问答准确率上显著优于传统 RAG，但首次响应时间较长。通过引入文档 Memory 缓存机制，热门文档的重复查询响应时间接近传统 RAG 水平，在准确率与效率之间取得合理平衡。考研/考公问答场景对响应延迟的容忍度高于实时搜索场景，该 trade-off 在实际应用中可接受。

---

## 七、时间规划建议

| 阶段 | 时间 | 任务 |
|---|---|---|
| **准备期** | 第1-2周 | 读懂 ReMemR1 代码（重点：memory_revisit.py、tf_idf_retriever.py）；搭环境 |
| **数据期** | 第3-4周 | 收集并整理知识库；构建评测集（至少200条QA）|
| **开发期** | 第5-8周 | 实现 Memory 管理模块；集成 RAG；构建 Prompt 模板；基础 Demo |
| **实验期** | 第9-11周 | 跑对比实验；调参；分析失败案例 |
| **写作期** | 第12-14周 | 写论文；制作答辩 PPT |
| **答辩准备** | 第15周 | 查重、修改、定稿 |

---

## 八、论文结构建议

```
第一章 绪论
  1.1 研究背景与意义（考研/考公咨询需求 + 长上下文 LLM 挑战）
  1.2 国内外研究现状（RAG、Memory Agent、ReMemR1等）
  1.3 本文研究内容与贡献
  1.4 论文结构

第二章 相关工作
  2.1 检索增强生成（RAG）综述
  2.2 LLM Agent 记忆机制综述
  2.3 Revisitable Memory 机制（重点介绍 ReMemR1）
  2.4 考研/考公智能问答现状

第三章 系统设计
  3.1 整体架构
  3.2 知识库构建
  3.3 Memory 管理模块设计（创新点1+2）
  3.4 时序感知知识管理（创新点3）
  3.5 Prompt 设计（借鉴 ReMemR1 的模板）

第四章 系统实现
  4.1 数据采集与预处理
  4.2 Memory 模块实现
  4.3 RAG 流程实现
  4.4 问答 Agent 实现
  4.5 系统界面

第五章 实验与分析
  5.1 实验设置（数据集、评测指标、对比系统）
  5.2 主实验结果
  5.3 消融实验（去掉 recall / 去掉时序管理的影响）
  5.4 案例分析
  5.5 错误分析

第六章 总结与展望
```

---

## 九、关键代码复用指引（来自本 Repo）

可以直接复用或参考的代码：

| 文件 | 复用内容 |
|---|---|
| `recurrent/impls/tf_idf_retriever.py` | TF-IDF 记忆检索，直接用于 recall 模块 |
| `recurrent/impls/memory_revisit.py` 中的 `TEMPLATE` | Prompt 模板设计参考 |
| `recurrent/impls/memory_revisit.py` 中的 `_parse_recall_query` | 解析 `<recall>` 标签的正则逻辑 |
| `recurrent/impls/memory_revisit.py` 中的 `_parse_update_memory` | 解析 `<update>` 标签 |
| `taskutils/memory_eval/utils/` | 评测工具参考 |

---

## 十、注意事项与风险点

1. **数据版权**：爬取数据只用于学术研究，不商用；优先使用官方公开文件（PDF直接下载）
2. **模型资源**：推荐用 Qwen2.5-7B-Instruct 本地部署（需1张A100/4090），或直接调用 API（DashScope/OpenAI 兼容接口）规避硬件瓶颈
3. **评测集质量**：评测集必须手工校验，不能全自动生成；50-100条高质量 > 1000条低质量
4. **与论文的关系**：论文里的 ReMemR1 是通过 RL 训练实现的；你的毕设是通过 Prompt Engineering 在应用层实现类似思想，两者定位不同，论文里要写清楚这个区别，避免被认为是简单复现
5. **答辩核心论点**：你的贡献是"将 Revisitable Memory 思想从长文档阅读理解场景扩展到多轮对话问答场景，并适配到中文考研/考公知识密集型应用"

---

*最后更新：2026-04-13*
