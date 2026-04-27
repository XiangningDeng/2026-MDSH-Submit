# Test 集新闻 URL 修复分工说明

目标：5 个人分别爬取 `data/test/news.tsv` 的不同区间，先各自生成 cache 和 report，最后统一合并 cache，再一次性更新 `data/test/news.tsv`。

注意：

- 每个人的命令都带 `--dry-run`，只会生成 cache/report，不会修改 `data/test/news.tsv`。
- 中途断了不用重来，重新运行同一条命令即可，脚本会读取已有 cache 并跳过已经处理过的标题。
- 不要删除自己的 `data/url_cache_part*.jsonl` 文件。
- `--sleep 3` 表示每次搜索之间暂停 3 秒，用来降低被搜索引擎限流的风险，最开始可以用1s节约时间。

## 分工命令

### 第 1 个人

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --start-line 1 \
  --end-line 24192 \
  --cache data/url_cache_part1.jsonl \
  --report data/url_report_part1.csv \
  --dry-run \
  --sleep 3
```

### 第 2 个人

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --start-line 24193 \
  --end-line 48384 \
  --cache data/url_cache_part2.jsonl \
  --report data/url_report_part2.csv \
  --dry-run \
  --sleep 3
```

### 第 3 个人

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --start-line 48385 \
  --end-line 72576 \
  --cache data/url_cache_part3.jsonl \
  --report data/url_report_part3.csv \
  --dry-run \
  --sleep 3
```

### 第 4 个人

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --start-line 72577 \
  --end-line 96768 \
  --cache data/url_cache_part4.jsonl \
  --report data/url_report_part4.csv \
  --dry-run \
  --sleep 3
```

### 第 5 个人

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --start-line 96769 \
  --end-line 120961 \
  --cache data/url_cache_part5.jsonl \
  --report data/url_report_part5.csv \
  --dry-run \
  --sleep 3
```

## 合并 cache

等 5 个人都完成后，把 5 个 cache 文件都放在 `data/` 目录下：

```text
data/url_cache_part1.jsonl
data/url_cache_part2.jsonl
data/url_cache_part3.jsonl
data/url_cache_part4.jsonl
data/url_cache_part5.jsonl
```

然后运行：

```bash
python - <<'PY'
import json
from pathlib import Path

merged = {}
for path in sorted(Path("data").glob("url_cache_part*.jsonl")):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row["title"]
            if title not in merged or row.get("status") == "found":
                merged[title] = row

with open("data/url_repair_cache_merged.jsonl", "w", encoding="utf-8") as f:
    for row in merged.values():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"merged {len(merged)} titles")
PY
```

这一步会生成：

```text
data/url_repair_cache_merged.jsonl
```

## 生成合并后的报告

如果只想先看合并结果，不修改 `news.tsv`，运行：

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --cache data/url_repair_cache_merged.jsonl \
  --report data/url_repair_report_merged.csv \
  --apply-cache-only \
  --dry-run
```

这一步会生成：

```text
data/url_repair_report_merged.csv
```

## 最终更新 test/news.tsv

确认报告没问题后，运行下面命令正式更新 `data/test/news.tsv`：

```bash
python tools/repair_news_urls.py \
  --news-files data/test/news.tsv \
  --cache data/url_repair_cache_merged.jsonl \
  --report data/url_repair_report_merged.csv \
  --apply-cache-only
```

这一步不会再联网搜索，只会把合并后的 cache 应用到 `data/test/news.tsv`。

脚本会自动备份原文件：

```text
data/test/news.tsv.bak
```
