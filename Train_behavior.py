import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.metrics import roc_auc_score
import os


train_path = r"C:\Users\zheng\OneDrive\桌面\Capstone\behaviors.tsv"
val_path = r"C:\Users\zheng\OneDrive\桌面\Capstone\val behaviors.tsv"


print("Train exists:", os.path.exists(train_path))
print("Val exists:", os.path.exists(val_path))


train_df = pd.read_csv(
    train_path,
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

val_df = pd.read_csv(
    val_path,
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

print("\nTrain shape:", train_df.shape)
print("Val shape:", val_df.shape)

print("\nTrain head:")
print(train_df.head())

print("\nVal head:")
print(val_df.head())

# ===== 4. 解析 impressions =====
def parse_impressions(impressions_str):
    news_ids = []
    labels = []

    for item in str(impressions_str).split():
        news_id, label = item.rsplit("-", 1)
        news_ids.append(news_id)
        labels.append(int(label))

    return news_ids, labels

# ===== 5. 用 train 统计点击次数 =====
news_click_count = Counter()

for impressions in train_df["impressions"]:
    for item in str(impressions).split():
        news_id, label = item.rsplit("-", 1)
        if label == "1":
            news_click_count[news_id] += 1

print("\nTop 10 popular news in train:")
print(news_click_count.most_common(10))

# ===== 6. 打分函数 =====
def score_by_popularity(news_ids, click_counter):
    return [click_counter.get(news_id, 0) for news_id in news_ids]

# ===== 7. 在 val 上生成 labels 和 scores =====
all_labels = []
all_scores = []

for impressions in val_df["impressions"]:
    news_ids, labels = parse_impressions(impressions)
    scores = score_by_popularity(news_ids, news_click_count)

    all_labels.append(labels)
    all_scores.append(scores)

print("\nNumber of validation impressions:", len(all_labels))

# ===== 8. 评估函数 =====
def mean_auc(all_labels, all_scores):
    aucs = []
    for labels, scores in zip(all_labels, all_scores):
        if len(set(labels)) < 2:
            continue
        aucs.append(roc_auc_score(labels, scores))
    return np.mean(aucs) if len(aucs) > 0 else 0.0

def mrr_score(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0

def mean_mrr(all_labels, all_scores):
    values = [mrr_score(labels, scores) for labels, scores in zip(all_labels, all_scores)]
    return np.mean(values) if len(values) > 0 else 0.0

def dcg_at_k(labels_sorted, k):
    dcg = 0.0
    for i in range(min(k, len(labels_sorted))):
        rel = labels_sorted[i]
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(labels, scores, k):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_labels = [labels[i] for i in order]

    dcg = dcg_at_k(ranked_labels, k)
    ideal_labels = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal_labels, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg

def mean_ndcg(all_labels, all_scores, k):
    values = [ndcg_at_k(labels, scores, k) for labels, scores in zip(all_labels, all_scores)]
    return np.mean(values) if len(values) > 0 else 0.0

# ===== 9. 计算指标 =====
auc = mean_auc(all_labels, all_scores)
mrr = mean_mrr(all_labels, all_scores)
ndcg5 = mean_ndcg(all_labels, all_scores, 5)
ndcg10 = mean_ndcg(all_labels, all_scores, 10)

print("\n=== Validation Results ===")
print(f"AUC: {auc:.4f}")
print(f"MRR: {mrr:.4f}")
print(f"nDCG@5: {ndcg5:.4f}")
print(f"nDCG@10: {ndcg10:.4f}")

# ===== 10. 打印几个验证样本 =====
print("\n=== Sample Predictions on Val ===")
for i in range(min(3, len(val_df))):
    news_ids, labels = parse_impressions(val_df.iloc[i]["impressions"])
    scores = score_by_popularity(news_ids, news_click_count)

    print(f"\nSample {i+1}")
    print("news_ids:", news_ids)
    print("labels:  ", labels)
    print("scores:  ", scores)

print("\n=== END ===")