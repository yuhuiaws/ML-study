# Text-only LLM SFT 训练数据预处理

> 本文档聚焦于 **text-only（非代码、非SQL）LLM 的 SFT（Supervised Fine-Tuning）训练数据预处理**。SFT 与 RFT 的数据预处理有区别，需分开讨论。全参数 SFT 和 PEFT（LoRA 及其变体）的数据预处理流程是一致的。

## 完整流程

SFT 训练数据预处理本质上是一个**迭代过程**：

```
开始
 │
 ├── 1. 收集数据
 ├── 2. 统一数据格式（标准化数据结构）
 ├── 3. 基础的数据清洗和过滤
 ├── 4. 初步的数据去重
 ├── 5. 进一步的数据清洗和质量过滤
 ├── 6. 精细的数据去重
 ├── 7. 数据分布评估
 │     └── 需要再次收集数据？── 是 ──→ 返回步骤 1
 │           │ 否
 ├── 8. 混合数据策略
 │     ├── 非闭源混合数据需执行步骤 1~6
 │     └── 混合比例未达预期？── 继续收集
 ├── 9. 数据打包和版本管理
 │
 结束
```

---

## 1. 收集数据

### 关键决策因素

**LoRA vs 全参数 SFT 决策树：**

| 判断条件 | 结果 |
|---------|------|
| 高质量样本是否充足？否 → | **LoRA**（样本少，不适合全参数 SFT） |
| 需要注入大量新知识或复杂任务？是 → | **全参数 SFT**（LoRA 表达力不足） |
| 需要多任务/多租户部署？是 → | **LoRA**（共享 base + 多 adapter 切换） |
| 预算是否充足？是 → | **全参数 SFT**（效果最优） |
| 预算是否充足？否 → | **LoRA**（预算有限 + 单任务） |

### Multi-task 全参数 SFT 的数据比例

- 对于 LoRA SFT，**不建议**单个 adapter 做 multi-task，每个任务一个 adapter
- Multi-task 全参数 SFT 不一定一次暴力 SFT 效果最好，**多阶段 SFT 效果可能更好**
- 核心原则：**避免数据不平衡导致的"任务遗忘"**

**常见采样方法：**

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| 均匀采样 | 每个任务等概率采样 | 简单，小任务不被淹没 |
| 按比例采样 | 按原始数据量比例 | 头部任务主导，尾部学习不充分 |
| **按温度采样** | `p_i = (n_i^(1/T)) / Σ(n_j^(1/T))`，T=2~5 | **最常用**（mT5、PaLM、GLM 等采用） |
| 基于任务难度/重要性加权 | 核心任务更高采样比例 | 需先定义任务优先级 |

其他策略：设置数据量上限（每任务 max 50k~100k）、小数据集多 epoch 重复采样、动态调整（监控各任务 loss）、分阶段训练、**验证集独立评估**。

### 样本数量参考

| 模型参数量 | 建议 SFT 样本数 | 典型 Token 数 | 训练 Epoch |
|-----------|---------------|-------------|-----------|
| 1-3B | 5k - 50k | 5M - 50M | 2-5 |
| 7-8B | 10k - 100k | 10M - 200M | 2-4 |
| 13-14B | 50k - 200k | 50M - 500M | 2-3 |
| 30-34B | 100k - 500k | 100M - 1B | 1-3 |
| 65-70B | 200k - 1M | 200M - 2B | 1-2 |

> 模型参数越大，能消费更多的样本"知识"，天花板更高。实际需求取决于任务复杂度、数据质量和目标效果。

### 样本难度比例

| 难度 | 比例 | 作用 |
|------|------|------|
| 简单 | 20~30% | 锚定基本输出格式和通用模式，稳定训练 |
| 中等 | 40~50% | 主力学习区间，模型提升最大的部分 |
| 困难 | 20~30% | 拉高能力上限，学习复杂推理和边界处理 |
| Corner case | 5~10% | 修补边界行为，防止极端输入下崩溃 |

> 困难样本和 corner case 不是越多越好，超过 40% 可能导致训练不收敛。困难样本的质量必须严格保证。

### 多语种策略

- 如果 base model 已支持目标语种，**优先一个模型训练多语种**
- 以目标业务的语种分布为锚点，根据 base model 语种强弱调整：

| base model 对该语种 | 调整策略 |
|-------------------|---------|
| 强（如 Qwen 对中文） | 可适当减少该语种比例，少量样本就能激活能力 |
| 弱（如 Llama 对日文） | 需要增加比例来补偿，否则效果差 |

- 用**验证集效果**来驱动最终比例

### 数据来源

- 公司内部积累的数据
- 业务专家标注（最高质量，成本最高）
- 众包平台（如 Amazon MTurk）标注
- 直接购买对齐的数据集
- 开源对齐数据集（如 QA 形式的试卷）
- **借助 SOTA LLM 生成对齐样本**（性价比最高）：Self-Instruct、Evol-Instruct、回译法、拒绝采样

### 核心原则

> **数据质量 > 数据数量**。LIMA 论文结论：仅 1000 条高质量数据就能训出优秀 SFT 模型。SFT 的 Scaling 规律不同于预训练——预训练靠量取胜，SFT 靠质取胜。**宁可花时间精标 5k 条，也不要粗标 50k 条。**

### 客户案例

**案例 1：垂域 single task 全参数 SFT（中英混杂）**
- 100 条专家样本 + 4W 条 GPT 生成样本（规则过滤 + 10% 专家抽检）
- 客户 task 样本量与开源中文指令样本 1:1 混合
- 效果：客户可以接受

**案例 2：垂域 single task 全参数 SFT（防灾难性遗忘）**
- 高质量对齐 QA 数据 20W+
- 与开源对齐样本 1:1 混合
- 效果：目标 QA 任务优于闭源模型，其他 task 基本不降

**案例 3：开放域 single task 全参数 SFT**
- 线上流量收集 100W+ 真实用户数据
- 渐进式消费：每次 20~30W 数据 full SFT，逐步迭代
- 效果：客户满意

---

## 2. 统一数据格式（标准化数据结构）

### 必要性

统一格式后，一套代码逻辑即可处理所有数据：

```python
# 统一为 {"instruction": ..., "response": ...} 格式
def clean_instruction_data(sample):
    if len(sample["instruction"]) < 10:
        return None
    if has_toxic_content(sample["response"]):
        return None
    return sample
```

### 关键原则

**数据中只存语义内容，把模板和特殊 token 的注入交给模型 tokenizer 的 `chat_template`**。这样同一份数据可以无修改地用于不同底座模型的微调。

输入数据示例：
```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "法国的首都是哪里？"},
    {"role": "assistant", "content": "法国的首都是巴黎。"}
  ]
}
```

应用 chat template 后渲染为：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
法国的首都是哪里？<|im_end|>
<|im_start|>assistant
法国的首都是巴黎。<|im_end|>
```

### 痛点与解决方案

不同数据源格式各异，转换脚本编写和维护繁琐。**推荐使用 Claude Code + Bedrock Claude Opus 来生成格式转换脚本**（实测效果很不错）。

> 本目录中的 `convert_daily_dialog.py` 就是一个转换脚本示例，将 DailyDialog 数据集转为 Nova Forge SFT 要求的 `bedrock-conversation-2024` 格式。

---

## 3. 基础的数据清洗和过滤

### 3.1 文本规范化（可选）

| 功能 | 默认 | 说明 |
|------|------|------|
| `fullwidth_to_halfwidth` | ON | 全角英文字母/数字 → 半角；保留中文标点（，。！？等）不转换 |
| `normalize_whitespace` | ON | `\r\n` → `\n`，制表符 → 空格，合并连续空格，3+ 连续换行 → 2 |
| `normalize_unicode_nfd_strip_accents` | OFF | NFD 分解 + 去掉口音符号（e.g. é → e） |
| `lowercase_english` | OFF | 仅对 ASCII 字母小写，中文不受影响 |
| `traditional_to_simplified` | OFF | 繁体 → 简体（依赖 OpenCC） |

**针对中英文的特别建议：**
- **全角/半角**：英文字符与数字始终转为半角
- **标点符号**：中英混排场景保留原始标点（只把英文字符和英文标点转为半角）
- **中文繁简**：除非需要同时支持简繁，否则统一为一种

### 3.2 基于规则的过滤（必须）

| 规则类型 | 具体实现 |
|---------|---------|
| **Language-based** | 用 langdetect 检测语言，丢弃非目标语言样本 |
| **Statistic-based** | 文本长度过短/过长、标点符号占比过高 → 丢弃 |
| **Keyword-based (element)** | 删除 URL、HTML 标签、BBCode 标签 |
| **Keyword-based (sentence)** | 含"关注/转发/点赞/subscribe"等广告词 → 丢弃该句 |
| **Policy-based** | 全大写句子 → 丢弃；纯数字句子 → 丢弃；短句以"登录/注册/login/sign up"开头 → 丢弃 |

> **先做文本规范化，然后再做基于规则的过滤。** 本目录中的 `basic_clean_LLM_SFT_data.py` 实现了上述清洗逻辑。

---

## 4. 初步的数据去重

### 去重的两个层面

| 层面 | 说明 | 执行阶段 |
|------|------|---------|
| **层面 1：基于样本相似性** | input + output 拼接后整体去重 | 本步骤（初步去重） |
| **层面 2：相似 input 保留最优 output** | 同一问题只保留质量最好的回答 | 步骤 6（精细去重） |

### 去重方法

| 方法 | 说明 |
|------|------|
| **精确去重（Exact Dedup）** | 对 input+output 拼接文本计算 hash，hash 相等则重复 |
| **基于 Overlap ratio** | 计算 token/n-gram/子串共享比例，超阈值则重复 |
| **基于 LSH（常用）** | MinHash LSH 或 SimHash LSH，SimHash 将文章映射为 64bit，海明距离 ≤ 3 则为重复 |

### 保留策略

给重复样本打分，保留得分最高的那条。常见评分维度：
- **对话完整性**（多轮场景优先保留轮次更多的）
- **回复质量/信息量**
- **时间新鲜度**
- **来源可信度**

### Tips

- 去重**不考虑**训练数据和混合数据之间的相似度
- 如果有 system 字段，去重时忽略它
- 多轮对话：把每个轮次去掉 role 后的 content 拼接在一起做去重

> 本目录中的 `basic_dedup_LLM_SFT_data.py` 实现了初步去重逻辑。

---

## 5. 进一步的数据清洗和质量过滤

### 借助模型过滤

**质量打分过滤（可选）：**
- 无数据敏感问题：直接用**闭源 LLM 打分**
  - 评估维度：Instruction Following、Helpfulness、Correctness、Harmlessness
- 数据敏感：训练一个分类器来过滤低分样本
- 成本控制：先采样 1-2k 条用强模型标注，训练质量分类器，再批量打分

**文本审核（必须）：** 涉黄、涉恐、涉暴、涉政、辱骂、性别歧视、种族歧视、语言暴力等

**PII 识别与去除（必须）：** 借助模型识别并去除个人敏感信息

### 多轮对话的特殊处理

多轮对话核心难点在于**轮次间的依赖关系**，不能只看单条 turn。

**Turn 级别过滤：**
- 空轮/极短轮（"好的"、"嗯"、"OK"）
- 重复轮（assistant 不同 turn 给出几乎相同回答）
- 角色错乱（role 标签不严格交替）
- 截断检测（末轮 response 不完整）

**对话级别过滤（多轮数据特有、最关键）：**

| 检查项 | 说明 | 方法 |
|--------|------|------|
| 指代一致性 | 后续轮次的"它"、"这个"是否有明确指代对象 | NLI 模型 / LLM 判断 |
| 话题漂移 | 对话是否突然跳到完全无关的话题 | 相邻 turn 语义相似度，突降则标记 |
| 上下文遗忘 | assistant 是否"忘记"前面提到的信息 | 提取前文实体/约束，检查后文是否违背 |
| 逻辑一致性 | assistant 前后回答是否自相矛盾 | NLI 模型检测 entailment/contradiction |

**对话推进性**：过滤"原地踏步"的对话；轮次过多（>10-15 轮）重点审查。

**对话结构修复（可选）：**

| 问题 | 修复方式 |
|------|---------|
| 末轮截断 | 裁剪到最后一个完整的 assistant turn |
| 中间有空轮 | 删除该空轮及之后的所有轮次，保留前半段 |
| 前 N 轮高质量，后面变差 | 只保留前 N 轮作为较短的多轮样本 |
| system prompt 缺失 | 根据对话内容补充合理的 system prompt |

> **宁可裁剪为高质量的前 5 轮，也不要保留带缺陷的完整 10 轮。** 本目录中的 `LLM_SFT_advanced_quality_filter.py` 实现了质量过滤逻辑。

---

## 6. 精细的数据去重

### 两阶段去重

1. **语义/embedding 相似度去重**：input+output 拼接后做 embedding，相似样本只保留质量最好的一条
2. **相似 input 聚类 + 最优 output 选择**：同一问题的多条不同回答只保留最好的

### 如何聚类相似 input

| 方法 | 适用规模 | 优点 | 成本 |
|------|---------|------|------|
| LSH | 百万级 | 速度快 | 低 |
| 基于 Embedding 聚类 | 十万级 | 去重效果更好 | 中 |

> 中英混合数据建议使用支持多语言的 embedding 模型（如 Bedrock 上的 Cohere Embedding）。input 相似度阈值建议 0.7 左右，太高会漏掉改写式重复。

### Output 质量评估

| 信号 | 描述 | 成本 |
|------|------|------|
| 样本长度 | 过短通常质量差（不是越长越好），过滤掉极短的 | 零 |
| 模型打分 | 用开源 SOTA 模型对单个 (input, output) 打分 | 中 |
| LLM as Judge | 用第一档闭源模型对同一 group 的多个 output 做对比打分 | 高，适合小规模 |
| 基于规则 | 拒绝回答、重复句子、含乱码的 output 直接淘汰 | 零 |

### 多轮对话的精细去重

- **input 聚类**：只把 user 的所有轮次拼接起来作为 input 进行聚类
- **output 质量评估**：保持 assistant 每轮回复列表，逐轮评分，最后加权

> 本目录中的 `advanced_dedup_LLM_SFT_data.py` 实现了精细去重逻辑。

---

## 7. 数据分布评估

### 评估维度

| 维度 | 评估方法 | 告警条件 |
|------|---------|---------|
| **语言分布** | 统计每种语言的样本数量和 token 总量 | 目标语言样本 < 总量 1% 且绝对数量 < 5k |
| **任务分布** | 统计各任务类型占比 | 单一任务 > 50% 或关键任务 < 2k 条 |
| **样本长度** | input/output 的 token 长度分布（P10~P99） | 应近似对数正态分布，覆盖 50~2k+ tokens |
| **难度分布** | 按回复复杂度分级 | 简单:中等:困难 ≈ 3:5:2 |

### 综合评估流程

```
对每个维度分别统计
    │
是否存在维度低于最低阈值？
    ├── Yes → 标记为"需补充"维度
    │           └── 能否通过上采样缓解？
    │               ├── Yes → 执行上采样
    │               └── No  → 启动定向数据收集
    └── No → 分布是否严重偏斜？
              ├── Yes → 下采样/重新平衡
              └── No  → 进入下一步
```

### Tips

- **自动化报告**：写脚本对清洗后数据集自动生成分布报告
- **设定硬门槛**：提前定义每个维度最低样本量和最大占比
- **交叉维度分析**：如"中文×数学推理"交叉后可能不达标
- **上采样优先于重新收集**：缺口小则上采样，缺口大则重新收集
- **预留验证集**：按相同比例抽出，确保验证集与训练集分布一致
- 若数据集无 metadata，可借助 **Bedrock Claude** 对每个样本标注语种、任务难度、任务类型

> 本目录中的 `data_distribution_eval.py` 实现了分布评估逻辑，输出 `distribution_report.json`。

---

## 8. 混合数据策略

### 核心决策：混合监督数据还是非监督数据？

- **理论角度**：混合预训练数据（无监督）更经典
- **工程角度**：基于 Chat/Instruct 模型微调时，混合对齐监督数据更方便、更常见
- 对通用知识保持要求很高，可以两者都混

### 混合比例参考

| 来源/实践 | SFT 数据占比 | 非监督数据占比 | 备注 |
|----------|------------|-------------|------|
| Llama 2 (Meta) | ~大部分 | 少量 pretrain replay | 官方未公开精确比例 |
| CodeLlama | SFT 主体 | ~5% pretrain replay | 用于保持通用能力 |
| InternLM 系列 | 80-90% | 10-20% | 混入高质量无监督语料 |
| 通用经验值（小规模 SFT） | 90-95% | 5-10% | SFT 数据量 < 100K 时 |
| 通用经验值（大规模 SFT） | 70-85% | 15-30% | SFT 数据量 > 500K 时 |
| 极端保守策略 | 50-60% | 40-50% | 特别在意基础能力保持 |

> 可以从 20% 开始，跑一组实验看通用能力退化情况，大多数 SFT 场景下 15%-25% 就够了。

### Tips

- **LoRA SFT 一般不需要混合数据**，冻结的 base model 权重会保留通用能力
- 非闭源混合数据需执行步骤 2~6，执行后重新检查比例是否达标
- 非监督数据选取：与目标能力相关的高质量文本（百科、教科书、代码语料等）

---

## 9. 数据打包和版本管理

### 处理步骤

1. **应用 chat template（可选）**：根据模型 apply chat template
2. **样本 pack（可选）**：多个短样本合并为长样本（用 special token 分隔），提升训练效率
   - 不同框架对 cross-sample attention mask 的支持不同（Megatron-LM 支持自定义 mask，HF Trainer 默认不支持）
3. **数据 tokenization（可选）**：大规模（>100k 样本）建议离线 tokenization
4. **生成最终格式（可选）**：Arrow/Parquet 格式，可考虑压缩

### 版本管理

推荐分层管理的目录结构：

```
dataset_v2.1.0/
├── config.yaml       # 数据集配置与处理参数
├── manifest.json     # 数据统计与元信息
├── data/
│   ├── train.parquet
│   └── eval.parquet
├── scripts/
│   └── preprocess.py  # 可复现的处理脚本
└── CHANGELOG.md       # 各版本变更记录
```

**最低限度要做到：**
- 每份数据集的原始文件有 SHA256 校验
- 处理脚本与数据集版本绑定（同一脚本 + 同一原始数据 = 可复现输出）
- 不要原地修改已产出的数据集，新版本用新目录

### 数据报告

训练前应生成数据报告：
- **基本统计**：总样本数/token 数、各数据源占比、长度分布（均值/P95/P99/最大值）、轮次分布
- **质量指标**：重复率、语言分布、空回复/异常样本数
- **Packing 统计**：packed 序列总数、平均子样本数、padding 比例、有效 token 利用率
- **处理链路**：完整处理步骤、每步过滤/丢弃数量、tokenizer 和 chat template 版本

> 本目录中的 `data_packing.py` 实现了数据打包逻辑。

---

## 本目录文件说明

### 核心处理脚本

| 文件 | 对应流程步骤 | 说明 |
|------|-----------|------|
| `convert_daily_dialog.py` | 步骤 2 | 将 DailyDialog 数据集转为 Nova Forge SFT 格式 |
| `basic_clean_LLM_SFT_data.py` | 步骤 3 | 基础数据清洗和规则过滤 |
| `basic_dedup_LLM_SFT_data.py` | 步骤 4 | 初步数据去重（精确去重 + MinHash LSH） |
| `LLM_SFT_advanced_quality_filter.py` | 步骤 5 | 进一步质量过滤（模型打分 + 文本审核 + PII） |
| `advanced_dedup_LLM_SFT_data.py` | 步骤 6 | 精细数据去重（语义去重 + 相似 input 聚类） |
| `data_distribution_eval.py` | 步骤 7 | 数据分布评估，输出分布报告 |
| `data_packing.py` | 步骤 9 | 数据打包（多样本合并 + tokenization） |

### 辅助脚本

| 文件 | 说明 |
|------|------|
| `generate_test_data.py` | 生成测试用的模拟数据 |
| `generate_zh_mixed_samples.py` | 生成中英混合测试样本 |
| `inject_duplicates.py` | 注入重复样本用于测试去重逻辑 |

### 测试脚本

| 文件 | 测试对象 |
|------|---------|
| `test_convert_daily_dialog.py` | `convert_daily_dialog.py` |
| `test_clean_daily_dialog.py` | 清洗后的 DailyDialog 数据验证 |
| `test_dedup_sft_data.py` | `basic_dedup_LLM_SFT_data.py` |
| `test_advanced_dedup_LLM_SFT_data.py` | `advanced_dedup_LLM_SFT_data.py` |
| `test_advanced_quality_filter.py` | `LLM_SFT_advanced_quality_filter.py` |
| `test_data_distribution_eval.py` | `data_distribution_eval.py` |
| `test_data_packing.py` | `data_packing.py` |

### 数据文件

| 文件 | 说明 |
|------|------|
| `daily-dialog.txt` | DailyDialog 原始数据 |
| `daily-dialog-cleaned.jsonl` | 清洗后的 DailyDialog 数据 |
| `daily-dialog-bedrock-all.jsonl` | 转为 Bedrock 格式的完整数据 |
| `daily-dialog-bedrock-cleaned.jsonl` | 转为 Bedrock 格式的清洗数据 |
| `zh_mixed_samples.jsonl` | 中英混合样本 |
| `zh_mixed_cleaned.jsonl` | 清洗后的中英混合样本 |
| `zh_mixed_deduped.jsonl` | 去重后的中英混合样本 |
| `zh_mixed_filtered.jsonl` | 质量过滤后的中英混合样本 |
| `zh_mixed_annotated.jsonl` | 标注后的中英混合样本 |
| `zh_mixed_advanced_deduped.jsonl` | 精细去重后的中英混合样本 |
| `distribution_report.json` | 数据分布评估报告 |
| `train_packed.parquet` | 打包后的训练数据 |

---

## 参考

- [LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206) — 1000 条高质量数据训 SFT 模型
- [Self-Instruct](https://arxiv.org/abs/2212.10560) — 用种子任务引导模型生成新 instruction + response
- [WizardLM (Evol-Instruct)](https://arxiv.org/abs/2304.12244) — 对已有 instruction 做复杂度演化
