# 第一轮任务分配（5人）

本文档定义第一轮 5 位成员的任务安排，基于本仓库当前的 MIND 数据。

## 第一轮目标

- 在同一数据切分上，让所有方法都能端到端跑通。
- 使用统一评估设置，产出可横向对比的结果。
- 建立清晰、可复现的第一轮 baseline。

## 任务范围：召回（Retrieval）与排序（Ranking/CTR）

- 召回型方法：基于热度/相似度信号生成或打分候选。
- 排序/CTR 方法：在 impression 级样本上学习点击概率，并据此排序。

本轮所有方法都在 MIND 的 impression 点击标签（`news_id-1` / `news_id-0`）上评估，使用同一验证协议。

## 可用数据

当前数据格式支持所有已分配方法：

- `data/train/news.tsv`
- `data/train/behaviors.tsv`
- `data/valid/news.tsv`
- `data/valid/behaviors.tsv`

`news.tsv` 提供新闻内容字段（`news_id`、category、title、abstract 等）。  
`behaviors.tsv` 提供用户行为日志与 impression 点击标签（`news_id-1` 或 `news_id-0`）。

## 成员任务

## 1) Member A - Popularity Baseline（召回型）

- 方法：基于 `train/behaviors.tsv` 统计新闻全局点击次数，对候选打分。
- 原因：最快完成数据链路与评估链路的 sanity check。
- 交付：可运行 notebook + 验证指标。

## 2) Member B - TF-IDF 内容基线（召回型）

- 方法：使用 `title`/`abstract` 构建 TF-IDF 向量，按与用户历史的相似度对候选打分。
- 原因：轻量、可解释的内容基线。
- 交付：可运行 notebook + 验证指标 + 简短预处理说明（tokenizer/文本清洗）。

## 3) Member C - Item-based Collaborative Filtering（召回型）

- 方法：基于 `train/behaviors.tsv` 的共点击关系构建 item-item 相似度，并按与用户历史点击新闻的相似度对候选排序。
- 原因：经典协同过滤 baseline，比 popularity 更强，且实现成本低。
- 交付：可运行 notebook + 指标 + 相似度选择说明（cosine/Jaccard）与 top-K 邻居设置。

## 4) Member D - Logistic Regression CTR（排序 / CTR）

- 方法：在 impression 级样本上做二分类点击预测，使用轻量表格特征。
- 建议特征：category、subcategory、历史长度、简单时间特征等。
- 交付：可运行 notebook + 指标 + 特征清单。

## 5) Member E - LightGBM CTR（排序 / CTR）

- 方法：与 Logistic Regression 使用同一套样本构造，模型升级为 LightGBM。
- 原因：非线性能力更强，工程代价仍较低。
- 交付：训练/推理结果 + 指标 + 特征重要性图或表。

## 统一 I/O 与评估规则

所有成员必须遵守以下规则，保证结果可比：

- 相同训练/验证切分：
  - train: `data/train/*`
  - validation: `data/valid/*`
- 相同随机种子：`42`
- 相同评估指标：
  - `AUC`
  - `MRR`
  - `nDCG@5`
  - `nDCG@10`
- 相同输出目录结构：
  - `outputs/<method_name>/metrics.json`
  - `outputs/<method_name>/prediction.txt`（可选但强烈建议）
  - `outputs/<method_name>/run.log`
  - `outputs/leaderboard_round1.csv`（全员统一汇总表）

### Shared preprocessing（共享样本展开）要求

- 本轮所有成员使用同一套 impression-candidate 展开逻辑（共享脚本/共享中间文件）。
- 已统一维护：
  - 统一展开脚本：`task_assignment/shared_preprocessing.py`
  - 统一评估脚本：`task_assignment/eval_round1.py`
  - 统一中间数据目录：`outputs/shared/`
    - `outputs/shared/train_candidates.csv`
    - `outputs/shared/train_candidates.parquet`
    - `outputs/shared/valid_candidates.csv`
    - `outputs/shared/valid_candidates.parquet`
- 建议先执行：
  - `python task_assignment/shared_preprocessing.py`

### Retrieval 方法限制

- 第一轮中，所有召回型方法必须对 `validation` 中每条 impression 已给定的候选新闻打分。
- 不允许从全量新闻库额外检索 top-K 来替代 validation 候选。

### 召回模型与 CTR 模型的公平对比

只有在输出对齐到同一 impression-candidate 粒度时，召回方法与 CTR 方法才能用同一套指标（`AUC`/`MRR`/`nDCG`）公平比较。

- 以下情况不能直接比较：
  - 召回只输出全库 top-K 列表；
  - CTR 输出的是候选级点击概率 `p(click)`。
- 公平比较需要统一记录格式：
  - `(impression_id, candidate_news_id, label, score)`
  - CTR 的 `score` 可为 `p(click)`；
  - 召回的 `score` 可为相似度/热度分（不要求是校准概率）。

只要两类方法都对同一 impression 候选集打分，就可以共用同一评估脚本，公平比较排序质量。

## Notebook 命名建议

- `baselines/popularity.ipynb`
- `baselines/tfidf.ipynb`
- `baselines/lr_ctr.ipynb`
- `baselines/lightgbm_ctr.ipynb`
- `baselines/itemcf.ipynb`


