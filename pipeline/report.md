# Pipeline Report

## Goal

The goal is to build a modular recommendation pipeline:

```text
Recall Module
-> LightGBM Ranking Model
-> TopK Recommendations
```

The pipeline supports two scenarios:

```text
Offline evaluation:
  evaluate ranking metrics on the full validation candidate set.

Online/backend-style serving:
  use recall models to generate or filter a smaller candidate pool,
  then use LightGBM to rerank the recalled candidates.
```

## Step 1: TF-IDF Recall + LightGBM Baseline

The first step was to build the initial end-to-end pipeline using TF-IDF as the first recall signal and LightGBM as the ranking model.

This step tested both offline evaluation and online-style filtering:

```text
Offline evaluation:
  full candidate set -> LightGBM ranking.

Online-style filtering:
  TF-IDF topK candidates -> LightGBM reranking.
```

Important evaluation detail:

```text
When recall cutoff is applied, candidates outside the recalled pool are still kept
in the final prediction file, but assigned score = -1e9.

This keeps the original validation impression structure and makes offline metrics
reflect both recall loss and ranking quality.
```

Full validation results:

| Run | Description | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Pipeline LightGBM baseline | full candidates, no recall feature, no cutoff | 0.6408 | **0.3539** | **0.3366** | **0.3962** |
| Pipeline LightGBM baseline + tuning | full candidates, tuned LightGBM | **0.6462** | 0.3482 | 0.3323 | 0.3925 |
| TF-IDF score + LightGBM | TF-IDF score as ranking feature | 0.6218 | 0.3455 | 0.3253 | 0.3849 |
| TF-IDF top20 + LightGBM | online-style cutoff | 0.5752 | 0.3229 | 0.3000 | 0.3580 |
| TF-IDF top50 + LightGBM | online-style cutoff | 0.5907 | 0.3368 | 0.3173 | 0.3741 |
| TF-IDF top100 + LightGBM | online-style cutoff | 0.6134 | 0.3433 | 0.3244 | 0.3837 |

Takeaway:

```text
TF-IDF was useful as a simple first recall baseline, but it did not improve the
ranking model when used directly as a feature.

TF-IDF cutoff also reduced ranking metrics because some clicked items were filtered out.
Larger topK performed better because it preserved more positive candidates.

Conclusion:
TF-IDF alone was not strong enough, so the next step was to test more recall sources.
```

## Step 2: Single Recall Source Ablation

The second step added four more recall sources and tested whether their raw scores could improve offline ranking.

New recall sources:

```text
Popularity
Category
ItemCF
EntityEmbedding
```

Each recall score was added into LightGBM as a ranking feature and compared against the LightGBM-only baseline.

50k impression ablation results:

| Run | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| LightGBM-only 50k | 0.5476 | 0.2946 | 0.2712 | 0.3357 |
| Popularity + LightGBM 50k | 0.5299 | 0.2887 | 0.2638 | 0.3283 |
| Category + LightGBM 50k | **0.5595** | **0.2956** | **0.2717** | **0.3365** |
| ItemCF + LightGBM 50k | 0.5150 | 0.2613 | 0.2375 | 0.2994 |
| EntityEmbedding + LightGBM 50k | 0.5512 | 0.2872 | 0.2639 | 0.3288 |

Because Category had the strongest 50k result, it was also tested on full validation:

```text
Category + LightGBM full:
  AUC = 0.6032
  MRR = 0.3343
  nDCG@5 = 0.3145
  nDCG@10 = 0.3730
```

Takeaway:

```text
Raw single recall scores did not outperform the original Pipeline LightGBM baseline.

Popularity and ItemCF were weak as ranking features.
Category had some signal, but still did not beat the full baseline.
EntityEmbedding slightly improved AUC in the 50k test but did not improve top-ranking metrics.

Conclusion:
Raw recall scores were not effective enough as direct ranking features.
The role of recall should shift toward backend candidate generation and filtering.
```

## Step 3: Recall-Stage Evaluation and Multi-Recall Union

The third step changed the focus from offline ranking features to backend-style candidate generation.

Instead of asking:

```text
Does this recall score improve LightGBM ranking?
```

this step evaluated:

```text
Can this recall method keep clicked items while reducing the candidate pool?
```

Recall-stage metrics:

```text
hit_rate:
  fraction of impressions where at least one clicked item is kept.

positive_keep_rate:
  fraction of all clicked candidate rows kept by recall.

candidate_keep_rate:
  fraction of candidate rows kept after recall filtering.

avg_candidates_kept_per_impression:
  average number of candidates kept per impression.
```

Important filtering rule:

```text
By default, recall evaluation only keeps candidates with score > 0.

Zero-score candidates are not used to fill topN because score = 0 usually means
the recall model has no useful matching signal.

Using zero-score candidates would be affected by the shuffled MIND candidate order and could artificially inflate recall coverage.
```

Single recall results at top100 on 50k impressions:

| Recall Source | Hit Rate | Positive Keep Rate | Avg Candidates |
|---|---:|---:|---:|
| TF-IDF | **0.8670** | **0.8488** | 29.82 |
| EntityEmbedding | 0.8590 | 0.8053 | 26.79 |
| Category | 0.7886 | 0.7613 | 23.67 |
| Popularity | 0.4541 | 0.3864 | 14.67 |
| ItemCF | 0.1823 | 0.1465 | 5.00 |

Multi-recall union results:

| Recall Union | topN | Positive Keep Rate | Avg Candidates |
|---|---:|---:|---:|
| 5-way union | 50 | **0.9521** | 33.64 |
| 3-way union | 50 | 0.9230 | 31.24 |
| 5-way union | 100 | **0.9711** | 35.97 |
| 3-way union | 100 | 0.9517 | 34.73 |

The selected hybrid recall combination was:

```text
TF-IDF + EntityEmbedding + Category
```

Takeaway:

```text
TF-IDF, EntityEmbedding, and Category were the strongest and most interpretable recall sources.

Popularity and ItemCF added limited coverage and were weaker individually, so they were not
prioritized in the main hybrid recall module.

Conclusion:
The pipeline moved from single recall scoring to multi-recall candidate generation.
The main hybrid recall module became TF-IDF + EntityEmbedding + Category.
```

## Step 4: Hybrid Recall for Backend Filtering and Offline Ranking Features

The fourth step used multi-recall in two different ways:

```text
1. Online/backend filtering:
   multi-recall topN union -> candidate pool -> LightGBM reranking.

2. Offline ranking features:
   no candidate cutoff;
   use multi-recall interaction features on the full candidate set.
```

For backend filtering, each recall source selects topN candidates with score > 0, then the selected candidates are unioned and deduplicated.

As in Step 1, candidates outside the recalled pool are assigned:

```text
score = -1e9
```

This preserves the full validation impression structure and makes metrics reflect recall loss.

### Backend Filtering Results

Hybrid recall filtering full validation results:

| Run | Candidate Reduction | Positive Keep Rate | Hit Rate | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3-way union@20 + LightGBM | **40.88%** | 81.32% | 89.97% | 0.5964 | 0.3328 | 0.3117 | 0.3705 |
| 3-way union@50 + LightGBM | 16.84% | 92.35% | 94.93% | 0.6219 | 0.3394 | 0.3209 | 0.3828 |
| 3-way union@100 + LightGBM | 7.61% | **95.21%** | **95.72%** | **0.6298** | **0.3433** | **0.3261** | **0.3860** |

Takeaway for backend filtering:

```text
union@100 achieved the best ranking metrics because it kept the most clicked items.

However, union@100 only reduced candidates by 7.61%, so its backend filtering value is limited.

union@50 is the better balanced setting:
it keeps 92.35% of positive items while reducing candidates by 16.84%.

Conclusion:
union@50 is the most balanced backend candidate generation setting.
```

### Offline Ranking Feature Results

For offline ranking, multi-recall interaction features were tested without candidate cutoff.

Main hybrid recall features:

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

Full validation ranking feature results:

| Run | Description | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Pipeline LightGBM baseline | full candidates, no recall features | 0.6408 | **0.3539** | **0.3366** | **0.3962** |
| Pipeline LightGBM baseline + tuning | full candidates, tuned | **0.6462** | 0.3482 | 0.3323 | 0.3925 |
| Hybrid recall features | full candidates, no cutoff | 0.6373 | 0.3523 | 0.3365 | 0.3943 |
| Hybrid recall features + tuning | full candidates, no cutoff, tuned | 0.6376 | 0.3518 | 0.3361 | 0.3940 |

Takeaway for offline ranking features:

```text
Hybrid recall interaction features were more useful than raw single recall scores.

However, they still did not clearly outperform the Pipeline LightGBM baseline.

The result is close to the baseline, but not enough to claim a ranking improvement.
```

### Final Takeaway and Next Steps

```text
Multi-recall is most useful for backend candidate generation and filtering.
For this purpose, union@50 is the best balanced setting.

For offline ranking, multi-recall interaction features are better than raw recall scores,
but they still do not significantly improve over the LightGBM baseline.

Next, stronger neural recall models should be explored, such as sentence embedding recall,
two-tower models, or dual encoders. These stronger recall signals can be used for both
backend filtering and ranking features.
```

## Appendix: Files and CLI Updates

This section is optional for the presentation, but useful if implementation details are asked.

```text
run_pipeline.py:
  added CLI options:
  --hybrid-recall-top-n
  --hybrid-recalls
  --hybrid-include-zero-score
  --use-hybrid-recall-features

pipeline/recall_hybrid.py:
  implements multi-recall topN union,
  candidate filtering,
  hybrid recall features,
  positive keep rate and hit rate stats.

pipeline/config.py:
  adds hybrid recall config,
  recall source validation,
  hybrid recall feature columns.

pipeline/feature_builder.py:
  merges hybrid recall features into ranking features.

pipeline/runner.py:
  connects hybrid recall filtering/features into the full pipeline,
  restores full validation predictions with -1e9 for unrecalled candidates,
  caches entity embedding recall scores.

pipeline/recall_entity_embedding.py:
  optimizes entity embedding scoring with numpy array assignment.

pipeline/PIPELINE_LOG.md:
  records the full union@20/@50/@100 results and final takeaways.
```
