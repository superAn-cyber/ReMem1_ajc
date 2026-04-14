# 毕设题目候选列表（控制工程方向 × LLM + Memory）

> 面向工业/企业场景，结合 ReMemR1 可回溯记忆机制，均有公开数据集可用。
> 按"落地难度从低到高"排序，推荐优先看前三个。

---

## 题目一：基于可回溯记忆的工业设备故障诊断问答系统 ⭐ 最推荐

**一句话描述**：维修人员用自然语言描述设备故障现象，系统在故障手册和历史案例库中检索，结合多轮对话记忆给出诊断建议。

**为什么适合你**：
- 故障诊断是控制工程的核心内容，天然懂业务
- 故障手册、维修规程通常几十到几百页，天然适合长文本+分块记忆
- 多跳推理需求强：故障现象在一处，根因在另一处，需要跨文档连接

**Memory 机制如何发挥作用**：
- 用户第一轮描述振动异常，第三轮问"是不是和轴承有关"，需要 recall 第一轮信息
- 维修手册中"第3章 振动" 和 "第7章 轴承磨损" 需要跨章节关联

**数据集**：

| 数据集 | 内容 | 链接 |
|---|---|---|
| CWRU 轴承数据集 | 凯斯西储大学轴承振动信号 + 故障标注，附官方文档说明 | https://engineering.case.edu/bearingdatacenter |
| MaintIE | 真实工业维修工单文本，含故障描述、维修动作、零部件标注 | https://github.com/nlp-tlp/maintie |
| PHM Society 数据集 | 历年 PHM 故障预测挑战赛数据，含设备描述文档 | https://phmsociety.org/data-challenge |
| NASA 预测性维护数据集 | 涡扇发动机、轴承等退化数据，含技术报告文本 | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository |

**补充知识库来源**：设备厂商公开手册（如西门子、ABB 官网的公开 PDF）、机械工程领域 Stack Exchange（https://engineering.stackexchange.com）

---

## 题目二：工业安全事故分析智能问答系统 ⭐ 数据最丰富

**一句话描述**：基于真实工业事故调查报告构建知识库，回答"某类事故的原因是什么""如何预防类似事故"等问题。

**为什么适合你**：
- 安全生产是工业控制的重要组成，控制工程背景直接相关
- 事故调查报告每份长达几十页，天然适合长文本分块记忆
- 纯文本数据，不需要处理传感器信号，技术栈更简单
- 数据最容易获取：政府机构强制公开

**Memory 机制如何发挥作用**：
- 一份事故报告包含背景、经过、原因、整改措施多个章节，跨章节多跳推理
- 用户追问"这类事故还发生过哪些"，需要 recall 历史对话中讨论过的案例

**数据集**：

| 数据集 | 内容 | 链接 |
|---|---|---|
| OSHA 事故报告 | 美国职业安全健康局公开的工厂事故调查报告，数万份，纯文本 | https://www.osha.gov/severeinjury |
| CSB 化工事故报告 | 美国化学品安全委员会调查报告，含详细技术分析，每份 20-100 页 | https://www.csb.gov/investigations |
| ARIA 工业事故数据库 | 法国工业事故数据库，含事故描述和分析文本 | https://www.aria.developpement-durable.gouv.fr |
| 应急管理部事故通报（中文）| 中国官方工矿商贸事故通报，公开文本 | https://www.mem.gov.cn/gk/sgcc |

**优势**：OSHA 和 CSB 数据完全免费、格式规范、数量充足，直接可用。

---

## 题目三：面向预测性维护的长文档智能问答助手

**一句话描述**：企业维护人员查询设备健康状态、剩余寿命预测、维护建议，系统结合设备手册和历史维护记录给出答案。

**为什么适合你**：
- 预测性维护（PdM）是工业控制的前沿应用
- 设备退化过程描述本身就是长文本，适合分块记忆
- 工业界真实需求，企业场景感强

**Memory 机制如何发挥作用**：
- 用户在第1轮告知设备型号和使用年限，后续轮次基于这些信息做个性化推荐
- 维护手册中润滑周期在第2章，更换标准在第8章，需要跨章节关联

**数据集**：

| 数据集 | 内容 | 链接 |
|---|---|---|
| NASA CMAPSS | 涡扇发动机退化仿真数据集，附技术文档，工业 PdM 标准数据集 | https://www.kaggle.com/datasets/behrad3d/nasa-cmaps |
| PRONOSTIA 轴承数据集 | 法国 FEMTO-ST 研究所轴承全寿命数据，含实验报告 | https://www.kaggle.com/datasets/alanderex/pronostia |
| Azure Predictive Maintenance | 微软公开的 PdM 模拟数据集，含设备日志和错误描述文本 | https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance |
| MaintIE（同题目一）| 维修工单文本，可作为历史维护记录知识库 | https://github.com/nlp-tlp/maintie |

---

## 题目四：化工过程控制知识问答系统（基于 Tennessee Eastman）

**一句话描述**：围绕经典化工过程控制场景构建问答系统，回答工艺参数异常分析、操作规程查询、报警处置建议等问题。

**为什么适合你**：
- Tennessee Eastman Process（TEP）是过程控制领域最经典的基准，控制工程专业必学
- 把已有的专业知识直接转化为毕设内容
- 工厂 DCS 操作规程本身就是长文本，天然适合 RAG + 记忆

**Memory 机制如何发挥作用**：
- 操作规程中"报警处置"章节需要引用"工艺参数说明"章节的上下文
- 多轮追问某个变量异常时，需要 recall 之前对话中确认的操作状态

**数据集**：

| 数据集 | 内容 | 链接 |
|---|---|---|
| TEP（Tennessee Eastman）仿真数据 | 经典化工过程的操作数据，含21种故障类型，有大量配套论文和文档 | https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset |
| TEP Extended（Rieth 版本）| 更大规模的 TEP 扩展版，含更多故障场景描述 | https://github.com/camaramm/tennessee-eastman-profBraatz-multivariate |
| UCI 化工数据集集合 | UCI 机器学习库中多个化工/过程控制数据集 | https://archive.ics.uci.edu/datasets?search=chemical |

**补充知识库**：TEP 原始论文（Downs & Vogel 1993）及后续综述文章（公开可下载）可直接作为知识库文本。

---

## 题目五：电力系统运维知识问答助手

**一句话描述**：针对电力调度、变电运维场景，回答设备操作规程、故障处理、负荷预测等专业问题。

**为什么适合你**：
- 电力系统是控制工程的典型应用领域
- 电网运维规程文件体量大（每份几十到几百页），适合长文本记忆

**Memory 机制如何发挥作用**：
- 调度规程中某设备的操作步骤分散在不同章节，需要跨章节 recall
- 用户分多轮查询同一次故障的不同方面，需要 session 级记忆

**数据集**：

| 数据集 | 内容 | 链接 |
|---|---|---|
| Open Power System Data | 欧洲电力系统公开数据，含负荷、发电量、价格等时序数据及说明文档 | https://open-power-system-data.org |
| IEEE PES 测试系统文档 | IEEE 14/30/118 节点标准测试系统，含完整技术文档 | https://labs.ece.uw.edu/pstca |
| ElectricityLoadDiagrams | UCI 电力负荷数据集，附使用说明 | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 |
| PowerGraph | 电力系统知识图谱数据集，含实体关系文本 | https://github.com/nlpie-research/PowerGraph |

---

## 题目六：制造业质量控制智能问答系统

**一句话描述**：工程师查询质量问题原因、检验标准、工艺参数调整建议，系统基于生产规程和历史质量报告给出答案。

**为什么适合你**：
- 质量控制涉及大量传感器监控与反馈，与控制工程直接相关
- 生产规程（SOP）文件量大，天然适合 RAG + 长文本记忆

**数据集**：

| 数据集 | 内容 | 链接 |
|---|---|---|
| SECOM 半导体制造数据 | 半导体生产过程传感器数据 + 质量标注，UCI 标准数据集 | https://archive.ics.uci.edu/dataset/179/secom |
| Steel Plates Faults | 钢板表面缺陷分类数据，UCI 数据集，含特征描述文档 | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| Bosch 生产线数据 | 博世生产线质量检测数据，含大量过程参数，Kaggle 竞赛数据集 | https://www.kaggle.com/c/bosch-production-line-performance |
| Manufacturing QA（中文）| Hugging Face 上的制造业知识问答数据集 | https://huggingface.co/datasets?search=manufacturing+qa |

---

## 题目七：KUKA 工业机器人运维知识问答系统

**一句话描述**：维护工程师通过自然语言查询 KUKA 机器人的故障代码含义、保养规程、编程问题、异常检测建议，系统基于官方手册和传感器异常记录给出答案。

**为什么适合你**：
- KUKA 是工业控制领域最主流的机器人品牌之一，控制工程专业背景直接对口
- KUKA 官方提供大量公开技术文档，是现成的高质量知识库
- 机器人运维手册动辄几百页，长文本 + 多章节跨段落推理，Memory 机制优势明显

**需要提前了解的实际情况**：
> 专门针对 KUKA 机器人的公开 ML 数据集数量较少，这是客观限制。但本题的核心知识库来源是官方文档（免费获取），传感器数据可用通用工业机器人数据集替代。这在毕设范围内完全合理，论文里说明清楚即可。

**Memory 机制如何发挥作用**：
- KUKA 错误代码手册、KRC 系统手册各自独立，跨手册关联需要 recall（如错误代码 → 对应维修章节）
- 用户第1轮描述关节异常，第3轮问"需要更换哪个零件"，需要 recall 之前的故障描述
- 多轮诊断对话中逐步缩小故障范围，session 记忆必不可少

**数据集与知识库来源**：

| 来源 | 内容 | 链接 | 说明 |
|---|---|---|---|
| **KUKA 官方文档中心** | KRC 控制器手册、机器人操作手册、错误代码大全，部分免费下载 | https://www.kuka.com/en-us/services/downloads | 注册即可下载，是最核心的知识库 |
| **KUKA Xpert 技术门户** | 产品技术文档、软件说明、配置指南，免费注册访问 | https://www.kuka.com/en-us/products/robotics-systems/software/cloud-software/kuka-xpert | 结构化文档，适合直接入库 |
| **Zenodo KUKA iiwa 多模态数据集** | KUKA LBR iiwa 机械臂的力、关节、音频传感器数据，含实验文档 | https://zenodo.org/records/6372438 | 可作为异常检测的传感器数据部分 |
| **Kaggle 工业机器人控制数据集** | 工业机器人控制系统传感器数据，含状态标注 | https://www.kaggle.com/datasets/ziya07/industrial-robot-control-system-dataset | 补充传感器类数据 |
| **Awesome Robotics Datasets** | 机器人领域公开数据集汇总索引，含多个 KUKA 相关条目 | https://github.com/mint-lab/awesome-robotics-datasets | 查找更多数据集的入口 |
| **UCI Robot Execution Failures** | 机器人执行失败场景数据集，含力传感器和操作状态描述 | https://archive.ics.uci.edu/dataset/138/robot+execution+failures | 通用机器人故障数据，可补充 |

**知识库构建建议**：
```
knowledge_base/
├── kuka_manuals/
│   ├── krc4_operating_manual.pdf      # KRC4 控制器操作手册（官网下载）
│   ├── error_code_reference.pdf       # 错误代码参考手册（官网下载）
│   └── maintenance_schedule.pdf       # 保养周期手册（官网下载）
├── sensor_data/
│   └── kuka_iiwa_multimodal/          # Zenodo 传感器数据
└── evaluation_set/
    ├── error_code_qa.json             # 错误代码含义问答（手工构建，100条）
    ├── maintenance_qa.json            # 保养规程问答（手工构建，100条）
    └── fault_diagnosis_qa.json        # 故障诊断多跳问答（手工构建，50条）
```

---

## 综合对比与推荐

| 题目 | 数据获取难度 | 与专业相关度 | Memory 机制适配度 | 工程实现难度 | 综合推荐 |
|---|---|---|---|---|---|
| 工业设备故障诊断 | ★★★★★ | ★★★★★ | ★★★★★ | ★★★ | **首选** |
| 工业安全事故分析 | ★★★★★ | ★★★★ | ★★★★★ | ★★ | **次选** |
| 预测性维护问答 | ★★★★ | ★★★★★ | ★★★★ | ★★★ | 推荐 |
| 化工过程控制 | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | 推荐（需专业背景）|
| 电力系统运维 | ★★★ | ★★★★ | ★★★ | ★★★ | 一般 |
| 制造业质量控制 | ★★★★ | ★★★ | ★★★ | ★★★ | 一般 |
| **KUKA 机器人运维** | ★★★ | ★★★★★ | ★★★★★ | ★★★★ | **推荐（有KUKA背景则首选）**|

**最终建议**：
- 如果想**稳妥好做**，选 **题目一（故障诊断）** 或 **题目二（安全事故）**，数据最充足，答辩也好讲
- 如果想**专业深度强**，选 **题目四（Tennessee Eastman）**，直接用专业课知识，和导师沟通也更顺畅
- 如果你**接触过 KUKA 机器人**（实验课、实习等），选 **题目七（KUKA 运维）**，专业背景直接转化为竞争优势，知识库用官方手册，数据稀缺问题可以在论文里合理说明
- 题目二的数据（OSHA/CSB 报告）**纯文本、量大、免费**，是最省力的数据来源

---

*最后更新：2026-04-14*
