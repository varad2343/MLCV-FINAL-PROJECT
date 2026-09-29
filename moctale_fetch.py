"""Fetch Moctale's publicly visible catalog and meter summaries.

This only requests public homepage/content pages. It does not log in, submit
forms, download poster images, or collect review text/user profiles.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.moctale.in/"
METER_LABELS = ("Perfection", "Go for it", "Timepass", "Skip")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def collect_homepage(soup: BeautifulSoup) -> list[dict[str, str]]:
    """Read public content links and their nearest preceding collection heading."""
    items: list[dict[str, str]] = []
    collection = "Featured"
    seen: set[str] = set()
    for element in soup.find_all(["h1", "h2", "h3", "a"]):
        if element.name in {"h1", "h2", "h3"}:
            heading = clean_text(element.get_text(" ", strip=True))
            if heading:
                collection = heading
            continue
        href = element.get("href", "")
        if "/content/" not in href:
            continue
        url = urljoin(BASE_URL, href)
        if url in seen:
            continue
        seen.add(url)
        title = clean_text(element.get_text(" ", strip=True))
        if title:
            items.append({"title_on_homepage": title, "collection": collection, "url": url})
    return items


def parse_detail(html: str, item: dict[str, str], scraped_at: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    joined = "\n".join(lines)
    h1 = soup.find("h1")
    title = clean_text(h1.get_text(" ", strip=True)) if h1 else item["title_on_homepage"]

    meta_match = re.search(
        r"\b(Movie|Show)\s*[•·]\s*(\d{4})(?:\s*[•·]\s*([^\n]+))?", joined, re.IGNORECASE
    )
    overview = ""
    if "Overview" in lines:
        start = lines.index("Overview") + 1
        stop = next((i for i in range(start, len(lines)) if lines[i] in {"Vibe Chart", "Cast", "Crew", "Moctale Meter"}), len(lines))
        overview = clean_text(" ".join(lines[start:stop]))

    meter = {}
    for label in METER_LABELS:
        match = re.search(rf"\b{re.escape(label)}\s*(\d{{1,3}})\s*%", joined, re.IGNORECASE)
        meter[f"meter_{label.lower().replace(' ', '_')}_pct"] = int(match.group(1)) if match else None

    vibe_start = lines.index("Vibe Chart") + 1 if "Vibe Chart" in lines else 0
    vibe_stop = next((i for i in range(vibe_start, len(lines)) if lines[i] in {"Cast", "Crew", "Moctale Meter", "User Reviews"}), len(lines))
    vibe_lines = lines[vibe_start:vibe_stop]
    vibe_pairs = []
    for index, line in enumerate(vibe_lines[:-1]):
        percent = re.fullmatch(r"(\d{1,3})%", vibe_lines[index + 1])
        if percent:
            vibe_pairs.append({"tag": line, "percentage": int(percent.group(1))})

    poster_url = ""
    images = soup.find_all("img")
    poster_candidates = [image for image in images if title.lower() in clean_text(image.get("alt", "")).lower()]
    if not poster_candidates:
        poster_candidates = [
            image for image in images
            if "media.moctale.in" in (image.get("src") or image.get("data-src") or "")
            and clean_text(image.get("alt", "")).lower() not in {"moctale", ""}
        ]
    for image in poster_candidates:
        src = image.get("src") or image.get("data-src") or image.get("data-lazy-src") or ""
        if src:
            poster_url = urljoin(BASE_URL, src)
            break

    record: dict[str, object] = {
        **item,
        "title": title,
        "content_type": meta_match.group(1).title() if meta_match else "",
        "release_year": int(meta_match.group(2)) if meta_match else None,
        "duration": clean_text(meta_match.group(3) or "") if meta_match else "",
        "overview": overview,
        "vibe_tags_json": json.dumps(vibe_pairs, ensure_ascii=False),
        "poster_url": poster_url,
        "scraped_at_utc": scraped_at,
        "page_status": "ok",
    }
    record.update(meter)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/moctale_public"))
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum public content pages to request (default: 20)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between content-page requests")
    args = parser.parse_args()
    if args.max_pages < 1 or args.delay < 0:
        parser.error("--max-pages must be positive and --delay must be non-negative")

    session = requests.Session()
    session.headers.update({"User-Agent": "MovieLens-Moctale-Research/1.0 (public catalog analysis)"})
    response = session.get(BASE_URL, timeout=30)
    response.raise_for_status()
    home_soup = BeautifulSoup(response.text, "html.parser")
    catalog = collect_homepage(home_soup)[: args.max_pages]
    if not catalog:
        raise RuntimeError("No public /content/ links were found on the Moctale homepage")

    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for item in catalog:
        if urlparse(item["url"]).netloc not in {"www.moctale.in", "moctale.in"}:
            continue
        try:
            page = session.get(item["url"], timeout=30)
            page.raise_for_status()
            records.append(parse_detail(page.text, item, scraped_at))
        except requests.RequestException as exc:
            records.append({**item, "scraped_at_utc": scraped_at, "page_status": f"error: {exc}"})
        time.sleep(args.delay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "moctale_public_content.csv", index=False)

    meter_cols = [f"meter_{label.lower().replace(' ', '_')}_pct" for label in METER_LABELS]
    available_meter_cols = [column for column in meter_cols if column in frame]
    if available_meter_cols:
        meter_long = frame.melt(
            id_vars=["title", "url", "collection", "scraped_at_utc"],
            value_vars=available_meter_cols,
            var_name="meter_category",
            value_name="displayed_percentage",
        ).dropna(subset=["displayed_percentage"])
        meter_long["meter_category"] = meter_long["meter_category"].str.removeprefix("meter_").str.removesuffix("_pct")
    else:
        meter_long = pd.DataFrame(columns=["title", "url", "collection", "scraped_at_utc", "meter_category", "displayed_percentage"])
    meter_long.to_csv(args.output_dir / "moctale_meter_patterns.csv", index=False)
    if not meter_long.empty:
        collection_summary = (
            meter_long.groupby(["collection", "meter_category"], dropna=False)["displayed_percentage"]
            .agg(mean_displayed_percentage="mean", content_pages="count")
            .reset_index()
        )
    else:
        collection_summary = pd.DataFrame(
            columns=["collection", "meter_category", "mean_displayed_percentage", "content_pages"]
        )
    collection_summary.to_csv(args.output_dir / "moctale_meter_summary_by_collection.csv", index=False)
    frame[["title", "collection", "url", "poster_url", "scraped_at_utc"]].to_csv(
        args.output_dir / "moctale_collections.csv", index=False
    )
    print(f"Saved {len(frame):,} public content-page records to {args.output_dir}")
    print(f"Meter category values found: {len(meter_long):,}; missing percentages remain blank.")


if __name__ == "__main__":
    main()
