# Pipeline Log

该文件是 pipeline 工作的持续更新日志

## Goal

搭建一个模块化的 recommendation pipeline，最终结构是一个 recall module 加一个 ranking model：

```text
Hybrid Recall Module
-> LightGBM Ranking Model (currently)
-> TopK recommendations
```

recall module 内部可以由多个 recall channels 组成。不同 recall 方法不是重复的，而是在捕捉不同的 user/news signal。

这个 pipeline 需要同时支持两种使用场景：

- Offline training / evaluation: 使用 multi-recall signals 作为 ranking features，在完整 candidate set 上评估。
- Online / backend deployment: 使用 multi-recall topN 生成 candidate pool，然后用 LightGBM reranking做展示。

## 2026-05-04 - Recall-stage Evaluation

本轮新增 `pipeline/evaluate_recall.py`，用于评估 recall-stage candidate generation 效果，而不是评估 LightGBM ranking metrics。

### Evaluation Metrics

```text
hit_rate:
  有多少 impression 至少保住一个 clicked item

positive_keep_rate:
  所有 clicked items 里，有多少被 recall topN 保留下来

candidate_keep_rate:
  recall 保留了多少 candidate rows

avg_candidates_kept_per_impression:
  平均每个 impression 保留多少 candidates

pairwise overlap / jaccard:
  不同 recall 之间的 candidate overlap，用于判断 complementarity
```

默认只评估 `score > 0` 的 candidates，不用 0 分 candidates 补满 topN。`score = 0` 通常代表该 recall 对这个 candidate 没有有效匹配信号；如果用 0 分补满 topN，会受到 MIND shuffled candidate order 的随机影响，可能虚高 recall coverage。

### Single Recall Results

50k recall-stage evaluation 显示，单路 recall 里表现最好的是：

```text
TF-IDF（最佳）
EntityEmbedding
Category
```

在 `top100` 下：

```text
TF-IDF:
  hit_rate = 0.8670
  positive_keep_rate = 0.8488
  avg_candidates = 29.82

EntityEmbedding:
  hit_rate = 0.8590
  positive_keep_rate = 0.8053
  avg_candidates = 26.79

Category:
  hit_rate = 0.7886
  positive_keep_rate = 0.7613
  avg_candidates = 23.67
```

`Popularity` 和 `ItemCF` 单路 coverage 明显较弱：

```text
Popularity top100:
  positive_keep_rate = 0.3864

ItemCF top100:
  positive_keep_rate = 0.1465
```

### Multi-recall Union Results

5路 union:

```text
TF-IDF + Popularity + Category + ItemCF + EntityEmbedding
```

3路 union:

```text
TF-IDF + EntityEmbedding + Category
```

50k 结果：

```text
5路 union@50:
  positive_keep_rate = 0.9521
  avg_candidates = 33.64

3路 union@50:
  positive_keep_rate = 0.9230
  avg_candidates = 31.24

5路 union@100:
  positive_keep_rate = 0.9711
  avg_candidates = 35.97

3路 union@100:
  positive_keep_rate = 0.9517
  avg_candidates = 34.73
```

### Current Takeaway

`Popularity` 和 `ItemCF` 单路 recall 较弱，但在 union 中能补到少量 clicked items。不过考虑 pipeline 结构简单性和模块可解释性，当前建议优先采用三路 recall：

```text
TF-IDF + EntityEmbedding + Category
```

当前阶段性方案：

```text
Current hybrid recall candidates:
  TF-IDF
  EntityEmbedding
  Category

暂不优先纳入:
  Popularity
  ItemCF
```

### Next Step

```text
1. 基于 TF-IDF + EntityEmbedding + Category 做 hybrid recall module。
2. 测试 union@50 / union@100 后接 LightGBM rerank。
3. 后续继续探索更强 neural recall，例如 two-tower / dual encoder / sentence embedding recall。
```

recall-stage evaluation 说明 multi-recall union 确实能显著提高 clicked item coverage；当前为了结构清晰，先选 `TF-IDF + EntityEmbedding + Category` 作为 hybrid recall 的主要组合。

## 2026-05-03 - Multi-recall Feature Ablation

本轮继续扩展 recall sources，并把每一路 recall score 接入 LightGBM features 再次对比评估，用于 offline training / evaluation。

### Files Added / Updated

新增 recall files：

```text
pipeline/recall_popularity.py
pipeline/recall_category.py
pipeline/recall_itemcf.py
pipeline/recall_entity_embedding.py
```

### CLI Update

CLI 改成更直观的使用方式：

```text
默认：LightGBM-only
需要哪个 recall score feature，就显式加哪个 --use-...-score
```

当前支持：

```text
--use-tfidf-score
--use-popularity-score
--use-category-score
--use-itemcf-score
--use-entity-embedding-score
```

### New Recall Features

Popularity:

```text
global_popularity_score
recent_popularity_score
category_popularity_score
popularity_score
recalled_by_popularity
```

Category:

```text
category_recall_score
subcategory_recall_score
category_recall_combined_score
recalled_by_category
```

ItemCF:

```text
itemcf_score
itemcf_rank
recalled_by_itemcf
```

EntityEmbedding:

```text
entity_embedding_score
entity_embedding_rank
recalled_by_entity_embedding
```

### 50k Ablation Results

下面结果都是 `50k train impressions + 50k valid impressions`，用于快速 smoke test / ablation，目的是先判断每个 recall feature 的潜在效果；除非特别说明，均为 no tune。

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| LightGBM-only 50k | 0.5476 | 0.2946 | 0.2712 | 0.3357 |
| Popularity + LightGBM 50k | 0.5299 | 0.2887 | 0.2638 | 0.3283 |
| Category + LightGBM 50k | **0.5595** | **0.2956** | **0.2717** | **0.3365** |
| ItemCF + LightGBM 50k | 0.5150 | 0.2613 | 0.2375 | 0.2994 |
| EntityEmbedding + LightGBM 50k | 0.5512 | 0.2872 | 0.2639 | 0.3288 |

鉴于Category + LightGBM 50k metrics均超过LightGBM-only 50k，补充 full result：

```text
Category + LightGBM full no tune:
  AUC = 0.6032
  MRR = 0.3343
  nDCG@5 = 0.3145
  nDCG@10 = 0.3730

Category + LightGBM full tune:
  same as no tune
```

### Current Takeaway

目前这 5 个 recall score 直接作为 LightGBM ranking feature，都没有超过原本 LightGBM baseline。

```text
TF-IDF: 负收益
Popularity: 负收益
Category: 有弱 signal，但 full 后没有超过 baseline
ItemCF: 负收益
EntityEmbedding: AUC 50k 小幅提升，但 MRR / nDCG 下降，feature importance 为 0
```

这说明当前这些 raw recall scores 不适合作为 LightGBM ranking feature 的主要提分方向。

但这不代表它们完全不能用于 recommendation pipeline。它们仍然可能适合做 online / backend candidate generation，所以下一步需要从 ranking-stage evaluation 转向 recall-stage evaluation。

### Next Step

下一步建议新增 recall-stage evaluation，重点看：

```text
top20 / top50 / top100 hit rate
positive keep rate
multi-recall overlap
multi-recall complementarity
```

也就是先判断每个 recall 在 candidate generation 阶段能不能抓到 clicked item，再决定是否放进最终 hybrid recall module。

还有接上一轮依旧：继续优化 LightGBM ranking model。

## 2026-05-02 - TF-IDF Recall + LightGBM Pipeline Baseline

现阶段先使用 TF-IDF 作为第一路 recall source，因为它是初期 recall baseline 里表现最好的方法。

### Files Added

```text
run_pipeline.py
pipeline/config.py
pipeline/data_prepare.py
pipeline/recall_tfidf.py
pipeline/feature_builder.py
pipeline/rank_lgbm.py
pipeline/metrics.py
pipeline/runner.py
pipeline/PIPELINE_LOG.md
```

### Supported CLI Options

```text
--mode eval / inference
--output-dir
--cache-dir
--top-k
--recall-top-k
--use-tfidf-score
--use-popularity-score
--use-category-score
--use-itemcf-score
--use-entity-embedding-score
--max-train-impressions
--max-valid-impressions
--tune-lgbm
```

重要说明：

- 默认情况下不加入任何 recall score features，也就是 LightGBM-only pipeline。需要哪个 recall feature，就显式加对应的 `--use-...-score`。
- `--recall-top-k K` 会启用 online-style TF-IDF recall cutoff，也就是先用 TF-IDF 每个 impression 保留 topK candidates，再交给 LightGBM reranking。
- `--use-tfidf-score` 会把 `tfidf_score` 加进 LightGBM features。其他 recall score 也是同样逻辑。
- 当前的 `inference` mode 还不是真正的 production serving。它目前仍然会训练并预测，只是不输出 metrics。

### Data Scale

valid impression 的 candidate count 是 long-tailed：

```text
mean: 37.47
median: 23
p75: 51
p90: 92
p95: 119
p99: 182
max: 295
```

这一点很重要，因为固定的 recall cutoff 会对 long-tail impressions 产生较强影响。比如 `recall-top-k=100` 对 candidate 数小于等于 100 的 impression 等于不筛选，但对 candidate 数超过 100 的 impression 会截断后面的 candidates。

后续解决办法：
短期：keep_n = max(x, y% candidates)或Ratio cutoff
长期：Recall union 合并去重

### Experiment Results

除非特别说明，下面结果都是 full validation results。

| Run | Description | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Original LightGBM baseline | LightGBM baseline | 0.6408 | **0.3539** | **0.3366** | **0.3962** |
| LightGBM-only via pipeline | Sanity check / ablation: `--tune-lgbm` | **0.6462** | 0.3482 | 0.3323 | 0.3925 |
| TF-IDF score + LightGBM | No cutoff, tuned LightGBM | 0.6218 | 0.3455 | 0.3253 | 0.3849 |
| TF-IDF recall top20 + LightGBM | Online-style cutoff, no tuning | 0.5752 | 0.3229 | 0.3000 | 0.3580 |
| TF-IDF recall top50 + LightGBM | Online-style cutoff, no tuning | 0.5907 | 0.3368 | 0.3173 | 0.3741 |
| TF-IDF recall top100 + LightGBM | Online-style cutoff, no tuning | 0.6134 | 0.3433 | 0.3244 | 0.3837 |
| TF-IDF recall top100 + tuned LightGBM | Online-style cutoff, tuned LightGBM | 0.6172 | 0.3438 | 0.3246 | 0.3840 |

### Takeaway: TF-IDF 的当前作用

实验结果显示，`tfidf_score` 直接作为 pipeline 的 recall 模型并没有带来metrics提升：

1. tfidf_score 不适合作为直接的 LightGBM ranking feature。
2. TF-IDF 更适合放在 recall 阶段，用来生成 candidate。
3. recall_top_k 越大，metrics 越高，因为更大的 topK 会减少 positive items 被 TF-IDF recall 筛掉的概率。

### Next Steps

1. 继续加入更强的 recall sources：
   - BM25
   - itemCF
   - embedding retrieval
   - popularity recall

2. Offline training / evaluation 时，把 multi-recall signals 作为 features：
   - `tfidf_score`
   - `itemcf_score`
   - `bm25_score`
   - `embedding_score`
   - `popularity_score`
   - `is_from_<recall>_topK`
   - `recall_source_count`

3. 继续优化 LightGBM ranking model：
   - 扩展 LightGBM search space
   - 尝试更多 ranking features
   - 做 feature ablation 判断每个 feature 是否有效
