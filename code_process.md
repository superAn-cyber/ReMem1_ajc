# ReMemR1 代码执行流程详解

整个项目分两大部分：**训练**和**推理/评测**。下面分别按执行顺序讲清楚。

---

## 一、核心思想（先理解这个，后面才能看懂代码）

**问题背景：** 给模型一篇很长的文章（几万字），让它回答问题。文章太长，模型一次性读不完。

**解决方法：** 把文章切成小块，一块一块地读，每读一块就把关键信息记下来（写入"记忆"）。读完所有块之后，根据积累的记忆回答问题。

**ReMemR1 的创新：** 普通的记忆只能往前看，ReMemR1 加了一个 **"回调（Callback）"** 机制——模型读当前块的时候，如果发现当前内容和之前某段记忆有关联，可以主动发出 `<recall>` 指令，检索并调出之前的记忆来辅助理解。

模型的输出格式长这样：

```
<thinking>我的思考过程...</thinking>
<update>更新后的记忆内容</update>
<recall>要检索的关键词</recall>   ← 可选，觉得需要回顾历史时才发出
```

---

## 二、推理流程（用训练好的模型回答问题）

> 入口：`taskutils/memory_eval/utils/rememr1.py` → `async_query_llm()`

### 第 1 步：准备输入

```
输入：一道问题 + 一篇很长的文章（context）
↓
把文章用 tokenizer 转成数字序列（token ids）
↓
如果文章超过最大长度，从中间截断，保留头尾
```

### 第 2 步：逐块处理文章（核心循环）

每次取 `chunk_size`（默认 5000）个 token，构建 prompt 发给模型：

```
┌─────────────────────────────────────────────────────────────┐
│  你面对一个问题、一段文章节选和之前的记忆。请按格式输出：      │
│                                                             │
│  <problem> {问题} </problem>                                │
│  <recalled_memory> {召回的历史记忆} </recalled_memory>       │
│  <memory> {上一轮的当前记忆} </memory>                       │
│  <section> {当前文章块} </section>                          │
│                                                             │
│  Updated memory:                                            │
└─────────────────────────────────────────────────────────────┘
```

模型收到后，生成回复。代码解析回复：

```python
memory       = 去掉 <recall>...</recall> 后剩余的文字  # 新记忆
recall_query = <recall> 里的内容                      # 要检索的关键词（可能没有）
```

如果有 `recall_query`，用 **TF-IDF 检索**（`tf_idf_retriever.py`）从历史所有记忆中找最相关的那条，作为下一轮的 `recalled_memory`。

> **TF-IDF 是什么：** 一种简单的文本相似度算法，不需要模型，只看词语重叠程度。知道它的作用是"根据关键词找最相关的历史记忆"就够了。

重复以上过程，直到文章所有块都处理完。

### 第 3 步：最终回答

所有块读完后，发最后一个 prompt：

```
┌──────────────────────────────────────────────────────┐
│  你面对一个问题和之前积累的记忆，请给出最终答案。        │
│                                                      │
│  <problem> {问题} </problem>                         │
│  <recalled_memory> {召回的记忆} </recalled_memory>    │
│  <memory> {最终记忆} </memory>                       │
│                                                      │
│  把答案放在 \boxed{} 里。                             │
└──────────────────────────────────────────────────────┘
```

模型输出最终答案，比如 `\boxed{巴黎}`。

### 整体推理流程图

```
文章
 │
 ├─ 块1 ──→ [模型] ──→ 记忆1，有无recall？──→ 检索历史
 │
 ├─ 块2 ──→ [模型] ──→ 记忆2，有无recall？──→ 检索历史
 │
 ├─ 块3 ──→ [模型] ──→ 记忆3 ...
 │
 └─ 所有块读完
          │
          └──→ [模型] ──→ 最终答案
```

---

## 三、训练流程（让模型学会这套行为）

> 入口：`scripts/1_run_train_ReMemR1_7B.sh` → `verl/trainer/main_ppo.py`

训练的目标：**让模型学会在正确时机发出 `<recall>`，并且最终能答对问题**。

用的方法是强化学习（RL），具体算法是 **GRPO**。

### 第 1 步：启动分布式训练环境

```bash
python3 -m verl.trainer.main_ppo \
    recurrent.memory.path="recurrent/impls/memory_revisit.py" \
    algorithm.adv_estimator=grpo \
    ...
```

用 **Ray** 管理多节点多 GPU，用 **Hydra** 管理所有超参数配置。

### 第 2 步：加载数据集

> `recurrent/impls/memory_revisit.py` → `MemoryDataset`

从 HotpotQA 数据集读取训练样本，每个样本包含：
- `prompt_ids`：问题的 token ids
- `context_ids`：文章的 token ids（已截断到最大长度）
- `context_length`：文章实际长度

### 第 3 步：Rollout（让模型"试答"）

> `recurrent/impls/memory_revisit.py` → `MemoryAgent`

这是训练中最关键的部分，执行和推理完全一样的逐块处理流程，但目的不同——这里是为了**收集模型的行为轨迹**，用于后续计算奖励。

`MemoryAgent` 实现了三个核心方法：

| 方法 | 作用 |
|---|---|
| `start()` | 初始化每个样本的记忆状态、历史记忆集合 |
| `action()` | 构建当前步的 prompt，返回给模型生成 |
| `update()` | 解析模型输出，更新记忆，执行 TF-IDF 检索 |

循环结构：
```
while not agent.done():
    messages = agent.action()      # 构建 prompt
    output   = model.generate()    # 模型生成
    agent.update(output)           # 解析输出，更新记忆
```

### 第 4 步：计算奖励

> `verl/trainer/ppo/core_algos.py` → `compute_grpo_outcome_advantage()`

#### 强化学习是什么

强化学习的核心思想是：**不告诉模型"应该输出什么"，而是让模型自己试，试对了就奖励，试错了就惩罚，模型逐渐学会做对的事情**。

类比：训练一只狗。你不会直接告诉它"把球叼回来的动作序列是这样的"，而是它做到了就给零食，做不到就不给，狗自然学会了。

对 LLM 来说：
- **行为** = 模型每一步输出的 token（文字）
- **奖励** = 最终答案对不对（EM 精确匹配得分 0 或 1）
- **目标** = 调整模型参数，让能答对的那类输出概率变大

#### 难点：奖励只有最后一步才有

问题在于，文章分成了很多块，模型要处理每一块才能最终回答。奖励（对不对）只有最后一步才知道，但前面每一步（写记忆、发 `<recall>`）也都影响了最终结果。

**怎么把最终的奖励"分配"回每一步？** 这就是强化学习里最核心的问题。

#### GRPO 的解法：组内比较

> 代码：`compute_grpo_outcome_advantage()`，配置 `ROLLOUT_N=16`

对**同一道题**，让模型生成 **16 条不同的轨迹**（通过随机采样，每次生成略有不同）。

```
同一道题，生成16次：
  轨迹1：chunk1记忆→chunk2发<recall>→...→最终答对  得分=1
  轨迹2：chunk1记忆→chunk2没发<recall>→...→最终答错  得分=0
  轨迹3：...→答对  得分=1
  ...
  16条轨迹的平均得分 = 0.5
```

然后对每条轨迹计算**优势值（Advantage）**：

```python
# 来自 core_algos.py 第 162 行
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

翻译成人话：
```
优势值 = (这条轨迹的得分 - 16条轨迹的平均得分) / 标准差

轨迹得分=1, 平均=0.5  →  优势值 > 0  →  这条轨迹的每个 token 都被"鼓励"
轨迹得分=0, 平均=0.5  →  优势值 < 0  →  这条轨迹的每个 token 都被"惩罚"
```

这样，即使是中间步骤（比如写记忆、发 `<recall>`），只要它所在的轨迹最终答对了，它就会得到正向强化。模型慢慢学会：**写好记忆、在合适时机发 `<recall>` → 更可能答对 → 应该多做**。

#### PPO Clip：防止更新幅度太大

> 代码：`compute_policy_loss()`，配置 `clip_ratio_high=0.20`

直接用优势值更新参数有个问题：一次更新太猛，模型会"走偏"，之后反而越来越差。

PPO 的解法是加一个"限速器"：

```python
# core_algos.py 第 419 行
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
```

`ratio` 是新模型和旧模型在这个 token 上概率的比值。`clamp` 把它限制在 `[0.8, 1.2]` 范围内（`clip_ratio_high=0.20`），意思是：**每次更新，模型行为的改变幅度不能超过 20%**。

#### KL 散度惩罚：防止忘掉原来的能力

> 代码：`kl_penalty()`，配置 `kl_loss_coef=0.001`

训练时还保留了一个"参考模型"（原始 Qwen，参数不更新）。每次更新时额外加一个惩罚项：

```
loss = -GRPO收益 + 0.001 × KL(当前模型 || 参考模型)
```

**KL 散度**衡量两个概率分布的差异。这个惩罚的意思是：当前模型和原始 Qwen 差太远了就会被惩罚，防止模型在学会 `<recall>` 的同时把原本的语言能力忘掉。

```python
# core_algos.py 第 501~503 行（low_var_kl 方式）
kl = ref_logprob - logprob
ratio = torch.exp(kl)
kld = (ratio - kl - 1)  # 这就是 KL 散度的数值估计
```

#### 完整的 loss 计算

```
最终 loss = PPO策略梯度loss + KL惩罚
          = -mean(clip(ratio, 0.8~1.2) × 优势值) + 0.001 × KL散度
```

梯度下降沿着让 loss 减小的方向更新参数，也就是让"优势值高的行为概率增大，优势值低的行为概率减小"。

### 第 5 步：更新模型参数

> `verl/trainer/ppo/ray_trainer.py`

用 **FSDP**（全分片数据并行）把 70 亿参数分散到多张 GPU 上，用 Adam 优化器沿 loss 梯度方向更新。

每次更新只用一小批数据（`ppo_mini_batch_size=8`），对同一批 rollout 数据重复更新多次，充分利用收集到的轨迹。

### 第 6 步：循环，保存 checkpoint

每隔 20 步保存一次模型，保留验证集上效果最好的版本。

整个训练循环：

```
第1轮：
  收集16×batch_size条轨迹（rollout）
  → 计算每条轨迹的优势值
  → 用 PPO+KL 更新参数
  → 保存 checkpoint

第2轮：用更新后的模型再收集轨迹...
...
第30轮（epoch）结束
```

随着训练进行，模型会逐渐学会：在关键信息处精准写入记忆、在需要关联前文时主动触发 `<recall>`，最终答题准确率持续上升。

---

## 四、代码文件速查表

| 文件 | 干什么的 |
|---|---|
| `scripts/1_run_train_ReMemR1_7B.sh` | 训练启动脚本，所有超参数在这里改 |
| `verl/trainer/main_ppo.py` | 训练 Python 入口 |
| `verl/trainer/ppo/ray_trainer.py` | 主训练循环（rollout → reward → update） |
| `verl/trainer/ppo/core_algos.py` | GRPO 算法实现，优势值计算 |
| `recurrent/interface.py` | RAgent/RDataset 的抽象基类（接口定义） |
| `recurrent/impls/memory_revisit.py` | ReMemR1 的具体实现（记忆、recall、数据集） |
| `recurrent/impls/tf_idf_retriever.py` | TF-IDF 历史记忆检索 |
| `taskutils/memory_eval/utils/rememr1.py` | 推理时的记忆循环逻辑（评测用） |
| `taskutils/memory_eval/run_eval.py` | 评测启动脚本，配置要测的模型 |
| `scripts/2_run_eval_ReMemR1.sh` | 评测入口 shell 脚本 |

---

## 五、训练 vs 推理的关系

```
训练阶段：
  MemoryAgent（memory_revisit.py）在 GPU 上跑 Rollout
  → 产生行为轨迹 → 计算奖励 → 更新模型权重
  → 保存为 checkpoint（即 ReMemR1-7B）

推理阶段：
  加载 ReMemR1-7B checkpoint，启动 sglang/vllm HTTP 服务
  rememr1.py 里的循环逻辑通过 HTTP 请求调用模型
  → 逐块处理文章 → 积累记忆 → 输出最终答案
```

训练时的 `MemoryAgent.action()/update()` 和推理时 `rememr1.py` 里的循环逻辑，**做的是完全相同的事情**，只是一个在 GPU 集群上批量跑，一个通过 HTTP 接口单条跑。
