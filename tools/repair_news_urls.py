#!/usr/bin/env python3
"""Repair MIND news.tsv URLs by searching current web URLs from article titles.

The original MIND assets.msn.com URLs now return PublicAccessNotPermitted. This
script searches each article title, chooses a likely current article URL, checks
that it is reachable, caches the mapping, and rewrites the URL column.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from requests import Session
from tqdm import tqdm


DEFAULT_NEWS_FILES = [
    Path("data/train/news.tsv"),
    Path("data/valid/news.tsv"),
    Path("data/test/news.tsv"),
]

BLOCKED_DOMAINS = {
    "assets.msn.com",
    "www.msn.com",
    "msn.com",
    "bing.com",
    "www.bing.com",
    "duckduckgo.com",
    "www.duckduckgo.com",
    "pinterest.com",
    "www.pinterest.com",
}

# DuckDuckGo's HTML endpoint currently serves results more consistently to a
# short browser-like UA than to a full Chrome UA.
USER_AGENT = "Mozilla/5.0"


@dataclass
class Candidate:
    title: str
    url: str
    snippet: str = ""
    source: str = ""


class DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[Candidate] = []
        self._capture_title = False
        self._capture_snippet = False
        self._current_url: str | None = None
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._capture_title = True
            self._current_url = clean_search_url(attr.get("href", ""))
            self._current_title = []
            self._current_snippet = []
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
            if self._current_url:
                self.results.append(
                    Candidate(
                        title=normalize_space("".join(self._current_title)),
                        url=self._current_url,
                        source="duckduckgo",
                    )
                )
        elif self._capture_snippet and tag in {"a", "div"}:
            self._capture_snippet = False
            if self.results:
                self.results[-1].snippet = normalize_space("".join(self._current_snippet))

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title.append(data)
        if self._capture_snippet:
            self._current_snippet.append(data)


class BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[Candidate] = []
        self._in_h2 = False
        self._capture_title = False
        self._current_url: str | None = None
        self._current_title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        if tag == "h2":
            self._in_h2 = True
        elif self._in_h2 and tag == "a" and attr.get("href", "").startswith(("http://", "https://")):
            self._capture_title = True
            self._current_url = clean_search_url(attr["href"])
            self._current_title = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
            if self._current_url:
                self.results.append(
                    Candidate(
                        title=normalize_space("".join(self._current_title)),
                        url=self._current_url,
                        source="bing",
                    )
                )
        elif tag == "h2":
            self._in_h2 = False

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title.append(data)


def normalize_space(text: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def comparable(text: str) -> str:
    text = html.unescape(text).lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def clean_search_url(url: str) -> str:
    url = html.unescape(url)
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
        wrapped = parse_qs(parsed.query).get("u", [""])[0]
        if wrapped.startswith("a1"):
            encoded = wrapped[2:]
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                return base64.urlsafe_b64decode(padded).decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                return url
    return url


def title_similarity(expected: str, found: str, snippet: str = "") -> float:
    expected_cmp = comparable(expected)
    found_cmp = comparable(found)
    snippet_cmp = comparable(snippet)
    if not expected_cmp or not found_cmp:
        return 0.0
    score = SequenceMatcher(None, expected_cmp, found_cmp).ratio()
    if expected_cmp in found_cmp or found_cmp in expected_cmp:
        score = max(score, 0.92)
    if snippet_cmp and expected_cmp in snippet_cmp:
        score = max(score, 0.9)
    return score


def url_looks_bad(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return True
    return host(url) in BLOCKED_DOMAINS


def search_duckduckgo(session: Session, title: str, timeout: float) -> list[Candidate]:
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(title)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    parser = DuckDuckGoParser()
    parser.feed(response.text)
    return parser.results


def search_bing(session: Session, title: str, timeout: float) -> list[Candidate]:
    url = "https://www.bing.com/search?q=" + quote_plus(title)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    parser = BingParser()
    parser.feed(response.text)
    return parser.results


def validate_url(session: Session, url: str, timeout: float) -> tuple[bool, str, int | None]:
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        status = response.status_code
        final_url = response.url
        content_type = response.headers.get("content-type", "")
        response.close()
        ok = status < 400 and "text/html" in content_type.lower()
        return ok, final_url, status
    except requests.RequestException:
        return False, url, None


def choose_url(
    session: Session,
    title: str,
    timeout: float,
    min_score: float,
    sleep_seconds: float,
    engines: Iterable[str],
) -> dict[str, object]:
    seen: set[str] = set()
    candidates: list[Candidate] = []
    errors: list[str] = []
    for engine in engines:
        try:
            if engine == "duckduckgo":
                candidates.extend(search_duckduckgo(session, title, timeout))
            elif engine == "bing":
                candidates.extend(search_bing(session, title, timeout))
        except requests.RequestException as exc:
            errors.append(f"{engine}: {exc}")
        if sleep_seconds:
            time.sleep(sleep_seconds)

    scored: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        if candidate.url in seen or url_looks_bad(candidate.url):
            continue
        seen.add(candidate.url)
        score = title_similarity(title, candidate.title, candidate.snippet)
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    for score, candidate in scored:
        if score < min_score:
            continue
        ok, final_url, status = validate_url(session, candidate.url, timeout)
        if ok:
            return {
                "status": "found",
                "url": final_url,
                "score": round(score, 4),
                "search_title": candidate.title,
                "source": candidate.source,
                "http_status": status,
            }

    return {
        "status": "missing",
        "url": "",
        "score": round(scored[0][0], 4) if scored else 0.0,
        "search_title": scored[0][1].title if scored else "",
        "source": scored[0][1].source if scored else "",
        "http_status": None,
        "errors": errors,
    }


def read_cache(path: Path) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["title"]] = row
    return cache


def append_cache(path: Path, title: str, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"title": title, **result}, ensure_ascii=False) + "\n")


def within_line_range(line_number: int, start_line: int, end_line: int) -> bool:
    if start_line and line_number < start_line:
        return False
    if end_line and line_number > end_line:
        return False
    return True


def load_news_rows(paths: list[Path], start_line: int = 0, end_line: int = 0) -> dict[str, str]:
    title_to_url: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                if not within_line_range(line_number, start_line, end_line):
                    continue
                if len(row) >= 6 and row[3]:
                    title_to_url.setdefault(row[3], row[5])
    return title_to_url


def rewrite_news_file(
    path: Path,
    replacements: dict[str, str],
    backup: bool,
    start_line: int = 0,
    end_line: int = 0,
) -> int:
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    changed = 0
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8", newline="") as src, temp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        for line_number, row in enumerate(reader, start=1):
            if len(row) >= 6 and within_line_range(line_number, start_line, end_line):
                new_url = replacements.get(row[3])
                if new_url and row[5] != new_url:
                    row[5] = new_url
                    changed += 1
            writer.writerow(row)
    os.replace(temp_path, path)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-files", nargs="*", type=Path, default=DEFAULT_NEWS_FILES)
    parser.add_argument("--cache", type=Path, default=Path("data/url_repair_cache.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/url_repair_report.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Only process N pending titles.")
    parser.add_argument("--start-line", type=int, default=0, help="First 1-based TSV line to process.")
    parser.add_argument("--end-line", type=int, default=0, help="Last 1-based TSV line to process, inclusive.")
    parser.add_argument("--retry-missing", action="store_true", help="Retry cached titles marked missing.")
    parser.add_argument("--apply-cache-only", action="store_true", help="Rewrite/report from cache without searching.")
    parser.add_argument("--dry-run", action="store_true", help="Search and report without rewriting TSV files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create news.tsv.bak files.")
    parser.add_argument("--min-score", type=float, default=0.82)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between search engine calls.")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["duckduckgo", "bing"],
        choices=["duckduckgo", "bing"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_line < 0 or args.end_line < 0:
        raise SystemExit("--start-line and --end-line must be >= 0")
    if args.start_line and args.end_line and args.start_line > args.end_line:
        raise SystemExit("--start-line cannot be greater than --end-line")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

    title_to_old_url = load_news_rows(args.news_files, args.start_line, args.end_line)
    cache = read_cache(args.cache)
    pending = []
    if not args.apply_cache_only:
        pending = [
            title
            for title in title_to_old_url
            if title not in cache or (args.retry_missing and cache[title].get("status") == "missing")
        ]
    if args.limit:
        pending = pending[: args.limit]

    for title in tqdm(pending, desc="Repairing URLs"):
        result = choose_url(
            session=session,
            title=title,
            timeout=args.timeout,
            min_score=args.min_score,
            sleep_seconds=args.sleep,
            engines=args.engines,
        )
        append_cache(args.cache, title, result)
        cache[title] = {"title": title, **result}

    replacements = {
        title: str(row["url"])
        for title, row in cache.items()
        if row.get("status") == "found" and row.get("url")
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["title", "status", "url", "score", "search_title", "source", "http_status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for title in title_to_old_url:
            row = cache.get(title, {"title": title, "status": "uncached", "url": ""})
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if args.dry_run:
        print(f"Dry run: {len(replacements)} titles have verified replacements.")
        print(f"Report: {args.report}")
        return 0

    total_changed = 0
    for path in args.news_files:
        if path.exists():
            changed = rewrite_news_file(
                path,
                replacements,
                backup=not args.no_backup,
                start_line=args.start_line,
                end_line=args.end_line,
            )
            print(f"{path}: updated {changed} rows")
            total_changed += changed

    found = sum(1 for row in cache.values() if row.get("status") == "found")
    missing = sum(1 for row in cache.values() if row.get("status") == "missing")
    print(f"Verified replacements: {found}; missing: {missing}; rows updated: {total_changed}")
    print(f"Cache: {args.cache}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
