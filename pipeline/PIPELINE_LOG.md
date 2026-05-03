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
--no-tfidf-score
--max-train-impressions
--max-valid-impressions
--tune-lgbm
```

重要说明：

- `--recall-top-k K` 会启用 online-style TF-IDF recall cutoff，也就是先用 TF-IDF 每个 impression 保留 topK candidates，再交给 LightGBM reranking。
- `--no-tfidf-score` 会从 LightGBM features 里移除 `tfidf_score`。这个主要用于 sanity check / ablation 效果对比。
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
| Original LightGBM baseline | LightGBM baseline | 0.6408 | 0.3539 | 0.3366 | 0.3962 |
| LightGBM-only via pipeline | Sanity check / ablation: `--no-tfidf-score --tune-lgbm` | 0.6462 | 0.3482 | 0.3323 | 0.3925 |
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
