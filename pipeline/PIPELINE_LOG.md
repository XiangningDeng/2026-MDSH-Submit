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

## 2026-05-12 - End-to-end Backend Pipeline and LambdaRank Trial

本轮主要把前面分开测试的 recall filtering 和 ranking features 串成更接近工业界 two-stage recommendation system 的完整 pipeline

### Current Pipeline

当前阶段性最终 pipeline：

```text
Recall:
  SentenceEmbedding + EntityEmbedding + Category union@50
  -> candidate pool
  -> recall score / rank / source signals

Feature generation:
  base features
  + hybrid recall features

Ranking:
  tuned binary LightGBM
  -> final ranked list
```

对应 CLI：

```bash
Microsoft/bin/python run_pipeline.py \
  --mode eval \
  --output-dir outputs/hybrid_recall_sentence_union50_features_full_tune \
  --cache-dir outputs/pipeline_cache \
  --hybrid-recall-top-n 50 \
  --hybrid-recalls sentence_embedding entity_embedding category \
  --use-hybrid-recall-features \
  --tune-lgbm
```

这个实验和前面的 full-candidate feature experiment 不同。这里先做 backend candidate filtering，再把 recall-derived features 交给 ranking model；因此它更像真实线上 pipeline，而不是只评估 ranking upper bound。

### End-to-end Backend Pipeline Result

对比对象是之前最好的 backend filtering setting：

```text
SentenceEmbedding + EntityEmbedding + Category union@50
-> base features
-> binary LightGBM
```

新的完整 pipeline 是：

```text
SentenceEmbedding + EntityEmbedding + Category union@50
-> base features + hybrid recall features
-> tuned binary LightGBM
```

对比涉及增加 hybrid recall features 和 tune ranking 模型，这两个模块。

Full validation results：

| Run | Candidate Reduction | Positive Keep Rate | Hit Rate | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| union@50 + base features + binary LightGBM | 16.62% | 93.23% | 95.62% | 0.6418 | 0.3462 | 0.3276 | 0.3897 |
| union@50 + hybrid features + tuned binary LightGBM | **16.62%** | **93.23%** | **95.62%** | **0.6439** | **0.3559** | **0.3404** | **0.3994** |

相较于之前 backend filtering 结果的提升：

```text
AUC:     0.6418 -> 0.6439 (↑)
MRR:     0.3462 -> 0.3559 (↑)
nDCG@5:  0.3276 -> 0.3404 (↑)
nDCG@10: 0.3897 -> 0.3994 (↑)
```

Filtering 相关指标保持不变，因为 candidate pool 没有变化：

```text
candidates_before = 2,740,998
candidates_after = 2,285,483
candidate_reduction = 16.62%
positive_keep_rate = 93.23%
hit_rate = 95.62%
avg_candidates_per_impression = 32.22
```

Feature importance 显示，reranker 确实在使用 recall-derived features：

```text
mean_recall_rank                #2
best_recall_rank                #3
sentence_embedding_rank         #4
recalled_by_sentence_embedding  #9
recalled_by_num_sources         #14
```

Takeaway:

```text
1. 在相同 union@50 candidate pool 下，加入 hybrid recall features 并 tune binary LightGBM 后，ranking metrics 全面提升。

2. 这说明 recall 不仅能用于 candidate generation，也能把 source / score / rank / overlap signals 作为 features 传给 ranking model。
```

### Comparison with Full-candidate Upper Bound

当前 full-candidate ranking upper bound 仍然是：

```text
Full candidate set
+ SentenceEmbedding / EntityEmbedding / Category hybrid features
+ tuned binary LightGBM
```

| Run | Filtering | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| full candidate + hybrid features + tuned binary LightGBM | no | **0.6446** | **0.3601** | **0.3432** | **0.4012** |
| union@50 + hybrid features + tuned binary LightGBM | yes | 0.6439 | 0.3559 | 0.3404 | 0.3994 |

Takeaway:

```text
完整 backend pipeline 在减少 16.62% candidates 的同时，ranking quality 已经非常接近 full-candidate tuned model。
```

### LambdaRank Trial

本轮也尝试把 ranking objective 从 binary classification 换成 LightGBMRanker / LambdaRank：

```text
--ranker lambdarank
```

LambdaRank 会按 impression_id 构造 group，并在训练前自动过滤无效 group：

```text
drop group size < 2
drop positive_count == 0
drop negative_count == 0
```

主要实验结果：

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| LambdaRank full candidate + hybrid features | 0.6148 | 0.3474 | 0.3287 | 0.3849 |
| LambdaRank union@50 + base features | 0.6003 | 0.3340 | 0.3136 | 0.3696 |
| LambdaRank union@50 + hybrid features | 0.6155 | 0.3440 | 0.3255 | 0.3824 |
| tuned binary LightGBM full candidate + hybrid features | **0.6446** | **0.3601** | **0.3432** | **0.4012** |
| tuned binary LightGBM union@50 + hybrid features | 0.6439 | 0.3559 | 0.3404 | 0.3994 |

Takeaway:

```text
1. 当前第一版 LambdaRank 没有超过 tuned binary LightGBM。即使加入 hybrid recall features，LambdaRank 的 ranking metrics 仍然明显低于 binary LightGBM。

2. 因此当前 pipeline 暂时继续使用 tuned binary LightGBM 作为 ranking model。LambdaRank 可以作为后续探索方向，但不是当前最佳方案。
```

### Current Takeaway

当前阶段性最终结论：

```text
1. 最像 production 的 end-to-end backend pipeline:
   SentenceEmbedding + EntityEmbedding + Category union@50
   + hybrid recall features
   + tuned binary LightGBM

2. 该 pipeline 在 full validation 上达到:
   candidate_reduction = 16.62%
   positive_keep_rate = 93.23%
   hit_rate = 95.62%
   AUC = 0.6439
   MRR = 0.3559
   nDCG@5 = 0.3404
   nDCG@10 = 0.3994

3. LambdaRank 已尝试，但当前不如 tuned binary LightGBM。
```

## 2026-05-11 - Sentence Embedding Hybrid Recall Full Evaluation

本轮把 lexical recall 中的 TF-IDF 替换成 neural sentence embedding recall，并重新测试新的三路 hybrid recall：

```text
SentenceEmbedding + EntityEmbedding + Category
```

SentenceEmbedding 使用 `sentence-transformers/all-MiniLM-L6-v2` 生成 news title / abstract embedding。每个 user impression 的 history embedding 取均值，然后和 candidate news embedding 做 cosine similarity，得到 `sentence_embedding_score` 和 `sentence_embedding_rank`。

### Files / CLI Updated

新增：

```text
pipeline/generate_sentence_embeddings.py
pipeline/recall_sentence_embedding.py
```

新增 / 扩展 CLI：

```text
--hybrid-recalls sentence_embedding entity_embedding category
--use-sentence-embedding-score
```

说明：

```text
generate_sentence_embeddings.py:
  在 Colab / GPU 环境下预先生成 sentence embedding cache。

recall_sentence_embedding.py:
  读取 embedding cache，计算 user-history-to-candidate 的 sentence embedding recall score。

--hybrid-recalls sentence_embedding entity_embedding category:
  用 SentenceEmbedding 替换 TF-IDF，和 EntityEmbedding / Category 组成三路 hybrid recall。

--use-sentence-embedding-score:
  不做 cutoff，只把 sentence embedding score 作为 LightGBM ranking feature。
```

### Why Replace TF-IDF

替换 TF-IDF 的原因是：SentenceEmbedding 和 TF-IDF 都是 content / text similarity recall，功能位最接近；而 EntityEmbedding 和 Category 捕捉的是不同信号，更适合作为互补 recall source 保留。

50k recall-stage single-source evaluation：

| Recall Source | topN | Hit Rate | Positive Keep Rate | Avg Candidates |
|---|---:|---:|---:|---:|
| TF-IDF | 100 | 86.70% | 84.88% | 29.82 |
| EntityEmbedding | 100 | 85.90% | 80.53% | 26.79 |
| Category | 100 | 78.86% | 76.13% | 23.67 |
| SentenceEmbedding | 20 | 82.42% | 69.73% | 14.63 |
| SentenceEmbedding | 50 | **91.99%** | **86.34%** | 25.00 |
| SentenceEmbedding | 100 | **94.25%** | **92.69%** | 31.76 |

Takeaway:

```text
SentenceEmbedding single recall 在 top50/top100 下明显超过 TF-IDF top100 的 hit_rate 和 positive_keep_rate。
因此这一轮不是把 EntityEmbedding / Category 换掉，而是用 SentenceEmbedding 替换原来的 TF-IDF lexical recall。
尝试过用 BM25 替换 TF-IDF ，但效果几乎和 TF-IDF 表现无异，所以尝试Sentence Embedding
```

### Backend Hybrid Filtering Results

```text
Old 3-way Recall: TF-IDF + Entity + Category
New 3-way Recall: Sentence Embedding + Entity + Category
```

Validation results（**加粗适用于同一个union级别的对比**）：

| Run | Candidate Reduction | Positive Keep Rate | Hit Rate | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old 3-way Recall union@20 | 40.88% | 81.32% | 89.97% | 0.5964 | 0.3328 | 0.3117 | 0.3705 |
| New 3-way Recall union@20 | **41.20%** | **82.34%** | **90.77%** | **0.5998** | 0.3296 | 0.3092 | 0.3686 |
| Old 3-way Recall union@50 | **16.84%** | 92.35% | 94.93% | 0.6219 | 0.3394 | 0.3209 | 0.3828 |
| New 3-way Recall union@50 | 16.62% | **93.23%** | **95.62%** | **0.6418** | **0.3462** | **0.3276** | **0.3897** |
| Old 3-way Recall union@100 | **7.61%** | 95.21% | 95.72% | 0.6298 | 0.3433 | 0.3261 | 0.3860 |
| New 3-way Recall union@100 | 6.65% | **95.93%** | **96.41%** | **0.6370** | **0.3459** | **0.3294** | **0.3888** |

当前最佳 backend candidate generation / filtering 结果：

```text
SentenceEmbedding + EntityEmbedding + Category union@50

Metrics:
  AUC = 0.6418
  MRR = 0.3462
  nDCG@5 = 0.3276
  nDCG@10 = 0.3897

Filtering:
  candidate_reduction = 16.62%
  positive_keep_rate = 93.23%
  hit_rate = 95.62%
```

旧版中更平衡的 backend filtering 设置是 `TF-IDF + EntityEmbedding + Category union@50`。保持 union@50 不变，只把 TF-IDF 替换成 SentenceEmbedding 后，各项 ranking metrics 都明显提升：

```text
AUC:     0.6219 -> 0.6418 (↑)
MRR:     0.3394 -> 0.3462 (↑)
nDCG@5:  0.3209 -> 0.3276 (↑)
nDCG@10: 0.3828 -> 0.3897 (↑)
```

新的 SentenceEmbedding 三路 hybrid recall 在不同 topN 下的取舍如下：

```text
union@20:
  candidate_reduction = 41.20%
  positive_keep_rate = 82.34%
  AUC = 0.5998
  过滤力度最强，但丢失了较多 clicked items。

union@50:
  candidate_reduction = 16.62%
  positive_keep_rate = 93.23%
  AUC = 0.6418
  这是目前最平衡的 backend candidate generation 设置。

union@100:
  candidate_reduction = 6.65%
  positive_keep_rate = 95.93%
  AUC = 0.6370
  保留的 clicked items 最多，但过滤效果较弱，并且 AUC 低于 union@50。
```

### Full Candidate Hybrid Feature Results

本轮也测试了新的三路 recall 不做 filtering，只作为 full-candidate ranking features：

```text
Full candidate set
+ SentenceEmbedding / EntityEmbedding / Category hybrid recall features
+ LightGBM
```

这条线的目的不是 backend filtering，而是 offline ranking metrics improvement。所有 valid candidates 都保留：

```text
candidates_before = 2,740,998
candidates_after = 2,740,998
candidate_keep_rate = 100%
```

Full validation results：

| Run | Description | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Pipeline LightGBM baseline | full candidate, no tune | 0.6408 | 0.3539 | 0.3366 | 0.3962 |
| Pipeline LightGBM baseline + tune | previous best full-candidate AUC | **0.6462** | 0.3482 | 0.3323 | 0.3925 |
| Old TF-IDF + Entity + Category hybrid features | previous hybrid-feature baseline | 0.6373 | 0.3523 | 0.3365 | 0.3943 |
| Old TF-IDF + Entity + Category hybrid features + tune | previous tuned hybrid-feature result | 0.6376 | 0.3518 | 0.3361 | 0.3940 |
| New Sentence + Entity + Category hybrid features | no tune | 0.6351 | 0.3544 | 0.3386 | 0.3962 |
| New Sentence + Entity + Category hybrid features + tune | best top-ranking result | 0.6446 | **0.3601** | **0.3432** | **0.4012** |

该设置下旧版最佳模型：

```text
Before SentenceEmbedding, the best full-candidate AUC result:

Metrics:
  AUC = 0.6462 (Pipeline LightGBM baseline + tune)
  MRR = 0.3518 (old 3-way recall with tune)
  nDCG@5 = 0.3361 (old 3-way recall with tune)
  nDCG@10 = 0.3940 (old 3-way recall with tune)
```

New best top-ranking result:

```text
SentenceEmbedding + EntityEmbedding + Category hybrid features + tune

Metrics:
  AUC = 0.6446
  MRR = 0.3601
  nDCG@5 = 0.3432
  nDCG@10 = 0.4012

Compared with the previous tuned LightGBM baseline:
  AUC is slightly lower: 0.6462 -> 0.6446 (↓)
  MRR is higher:         0.3482 -> 0.3601 (↑)
  nDCG@5 is higher:      0.3323 -> 0.3432 (↑)
  nDCG@10 is higher:     0.3925 -> 0.4012 (↑)

Compared with the old TF-IDF + EntityEmbedding + Category hybrid features + tune:
  AUC improves:     0.6376 -> 0.6446 (↑)
  MRR improves:     0.3518 -> 0.3601 (↑)
  nDCG@5 improves:  0.3361 -> 0.3432 (↑)
  nDCG@10 improves: 0.3940 -> 0.4012 (↑)
```

### Current Takeaway

当前最重要结论：

```text
1. Backend filtering / candidate generation:
   SentenceEmbedding + EntityEmbedding + Category union@50
   是目前最好的 backend candidate generation 设置。

   它在 full validation 上达到:
     AUC = 0.6418
     candidate_reduction = 16.62%
     positive_keep_rate = 93.23%
     hit_rate = 95.62%

   相比旧的 TF-IDF + EntityEmbedding + Category union@50，
   ranking metrics 全面提升。

2. Offline full-candidate ranking:
   SentenceEmbedding + EntityEmbedding + Category hybrid features + tune
   是目前最好的 top-ranking result。

   它在 full candidate set 上达到:
     AUC = 0.6446
     MRR = 0.3601
     nDCG@5 = 0.3432
     nDCG@10 = 0.4012

   它没有超过 tuned LightGBM baseline 的 AUC = 0.6462，
   但明显超过了该 baseline 的 MRR / nDCG@5 / nDCG@10。

3. Overall:
   SentenceEmbedding 不仅是比 TF-IDF 更强的 recall replacement，
   也能作为 hybrid recall interaction features 帮助 LightGBM 提升 top-ranking quality。
```

后续建议：

```text
1. 如果继续做模型升级，下一步可以尝试更贴近 recommendation objective 的 neural recall，
   例如 two-tower / dual encoder。
2. 尝试继续优化 ranking 模型：LightGBM
```

## 2026-05-05 - Hybrid Recall Filtering vs Multi-recall Features

本轮主要区分了 multi-recall 的两种接法：

```text
1. multi-recall as filtering:
   先用多路 recall topN union 筛 candidate，再交给 LightGBM rerank。

2. multi-recall as ranking features:
   不筛 candidate，在完整 candidate set 上加入 multi-recall interaction features，再交给 LightGBM ranking。
```

### Files / CLI Updated

新增：

```text
pipeline/recall_hybrid.py
```

新增 CLI：

```text
--hybrid-recall-top-n
--hybrid-recalls
--hybrid-include-zero-score
--use-hybrid-recall-features
```

说明：

```text
--hybrid-recall-top-n / --hybrid-recalls:
  用 multi-recall 做 candidate generation / filtering。

--use-hybrid-recall-features:
  不做 cutoff，只把 multi-recall interaction features 加入 LightGBM。
```

当前使用的三路 recall：

```text
TF-IDF + EntityEmbedding + Category
```

### A. Multi-recall Filtering Results

下面结果都是 full validation results，不加入 recall features，只测试：

```text
TF-IDF / EntityEmbedding / Category
-> 每路 topN
-> union 去重
-> LightGBM rerank
```

| Run | Candidate Reduction | Positive Keep Rate | Hit Rate | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3路 union@20 + LightGBM | **40.88%** | 81.32% | 89.97% | 0.5964 | 0.3328 | 0.3117 | 0.3705 |
| 3路 union@50 + LightGBM | 16.84% | 92.35% | 94.93% | 0.6219 | 0.3394 | 0.3209 | 0.3828 |
| 3路 union@100 + LightGBM | 7.61% | **95.21%** | **95.72%** | **0.6298** | **0.3433** | **0.3261** | **0.3860** |

Takeaway:

```text
在 full validation 上，ranking metrics 随 topN 增大而上升。
全量下 positive coverage 的影响更明显，union@100 保留最多 clicked items，
因此 ranking metrics 最高。

但 union@100 只减少 7.61% candidates，candidate reduction 很弱；
union@50 减少 16.84% candidates，同时保留 92.35% clicked items，
是更平衡的 backend candidate generation 设置；
union@20 reduction 最强，但 full ranking metrics 损失也最明显。
```

和 full-candidate LightGBM baseline 相比：

```text
Pipeline LightGBM baseline no cutoff no tune:
  AUC = 0.6408
  MRR = 0.3539
  nDCG@5 = 0.3366
  nDCG@10 = 0.3962

Best full hybrid filtering result union@100:
  AUC = 0.6298
  MRR = 0.3433
  nDCG@5 = 0.3261
  nDCG@10 = 0.3860
```

因此，full-candidate offline ranking 结果仍然优先报告 LightGBM baseline / hybrid recall features；hybrid filtering 更适合作为 backend candidate generation trade-off 展示。

### B. Multi-recall Features Results

本轮更重要的发现是：raw single recall score 效果不好，但 multi-recall interaction features 有明显价值。

新增的主要 features 包括：

```text
recalled_by_tfidf
recalled_by_category
recalled_by_entity_embedding
recalled_by_num_sources
recall_source_overlap_count
max_recall_score
mean_recall_score
tfidf_rank
category_rank
entity_embedding_rank
best_recall_rank
mean_recall_rank
```

Full validation results：

| Run | Description | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Pipeline LightGBM baseline | full, no recall features, no cutoff, no tune | 0.6408 | **0.3539** | **0.3366** | **0.3962** |
| Pipeline LightGBM baseline + tune | full, no recall features, no cutoff, `--tune-lgbm` | **0.6462** | 0.3482 | 0.3323 | 0.3925 |
| Hybrid recall features | full, no cutoff, no tune | 0.6373 | 0.3523 | 0.3365 | 0.3943 |
| Hybrid recall features + tune | full, no cutoff, `--tune-lgbm` | 0.6376 | 0.3518 | 0.3361 | 0.3940 |

Takeaway:

```text
Hybrid recall features 没有超过 Pipeline LightGBM baseline 的最高 AUC，
整体 metrics 和 Pipeline LightGBM baseline no tune 非常接近。

同时，和 Pipeline LightGBM baseline + tune 相比，
Hybrid recall features 的 MRR / nDCG@5 / nDCG@10 更高。
这说明 multi-recall interaction features 对 top-ranking metrics 有帮助，
也比直接加入 raw single recall score 更有效。
```

Feature importance 中稳定靠前的 multi-recall features：

```text
mean_recall_rank
best_recall_rank
recalled_by_num_sources
recalled_by_tfidf
recall_source_overlap_count
mean_recall_score
```

### Current Takeaway

当前最重要结论：

```text
1. multi-recall filtering:
   能展示 candidate reduction 和 positive_keep_rate 的 trade-off，
   full validation 上 union@100 ranking metrics 最高，
   但 candidate reduction 很弱；union@50 是更平衡的 backend candidate generation 设置。

2. multi-recall features:
   是目前比 raw single recall score 更有效的 ranking signal。
   在 full candidates 上不丢 positives，整体表现接近 Pipeline LightGBM baseline，
   并且相对 tuned baseline 提升了 MRR / nDCG@5 / nDCG@10。
```

后续建议：

```text
1. 报告 offline ranking result 时，重点使用 multi-recall features + LightGBM no cutoff。
2. backend / inference 时，使用 multi-recall topN candidate generation + LightGBM rerank；
   当前优先展示 union@50 的平衡版本，也可同时报告 union@100 的最高 ranking metrics。
3. 后续继续尝试更强 recall source，例如 BM25、sentence embedding、two-tower / dual encoder。
```

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

- 默认情况下不加入任何 recall score features，也就是 Pipeline LightGBM baseline。需要哪个 recall feature，就显式加对应的 `--use-...-score`。
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
| Pipeline LightGBM baseline | full, no recall features, no cutoff, no tune | 0.6408 | **0.3539** | **0.3366** | **0.3962** |
| Pipeline LightGBM baseline + tune | full, no recall features, no cutoff, `--tune-lgbm` | **0.6462** | 0.3482 | 0.3323 | 0.3925 |
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
