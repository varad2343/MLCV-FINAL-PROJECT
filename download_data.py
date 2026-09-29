"""Download larger MovieLens datasets and poster images for offline analysis.

  python download_data.py                        # MovieLens 25M -> data/ml-25m/  (262 MB zip)
  python download_data.py --dataset ml-32m       # MovieLens 32M -> data/ml-32m/  (239 MB zip)
  python download_data.py --posters 1000         # poster images for the 1,000 most-rated films

Datasets come from files.grouplens.org (F. M. Harper and J. A. Konstan, 2015,
The MovieLens Datasets: History and Context, ACM TiiS 5(4)); each zip is checked
against the MD5 GroupLens publishes. Downloads resume if interrupted.

Posters are the lead images of the films' English Wikipedia articles, saved to
data/posters/<movieId>.<ext> with a manifest. Most are copyrighted and hosted
by Wikipedia under fair use: keep them for private analysis and don't publish
or commit them (data/posters/ is git-ignored). Requests go one at a time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POSTERS = DATA / "posters"
BASE_URL = "https://files.grouplens.org/datasets/movielens/"
DATASETS = {
    "ml-25m": "25M ratings, 62,423 films, 162,541 users, tag genome",
    "ml-32m": "32M ratings, 87,585 films, 200,948 users",
    "ml-latest-small": "100k ratings, 9,742 films, 610 users (already in data/)",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def published_md5(session: requests.Session, name: str) -> str | None:
    """GroupLens publishes md5sum output: '<32 hex digits>  <file name>'."""
    try:
        text = session.get(f"{BASE_URL}{name}.zip.md5", timeout=30).text
    except requests.RequestException:
        return None
    match = re.search(r"\b[0-9a-fA-F]{32}\b", text)
    return match.group(0).lower() if match else None


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(session: requests.Session, url: str, destination: Path) -> None:
    """Stream to <destination>.part, resuming from where a previous run stopped."""
    partial = destination.with_suffix(destination.suffix + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with session.get(url, headers=headers, stream=True, timeout=60) as response:
        if response.status_code == 416:  # already complete
            partial.replace(destination)
            return
        response.raise_for_status()
        if done and response.status_code != 206:
            done = 0  # server ignored the range; start over
        total = done + int(response.headers.get("Content-Length", 0))
        mode = "ab" if done else "wb"
        next_report, started = 0.0, time.time()
        with open(partial, mode) as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    rate = done / max(time.time() - started, 1e-6) / 1e6
                    log(f"  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({done / total:4.0%}, {rate:.1f} MB/s)")
                    next_report += 0.1
    partial.replace(destination)


def fetch_dataset(name: str, keep_zip: bool) -> Path:
    target = DATA / name
    if (target / "ratings.csv").is_file() and (target / "movies.csv").is_file():
        log(f"{name} is already in {target.relative_to(ROOT)}")
        return target
    session = requests.Session()
    zip_path = DATA / f"{name}.zip"
    if not zip_path.is_file():
        log(f"Downloading {name}.zip from {BASE_URL}")
        download(session, f"{BASE_URL}{name}.zip", zip_path)
    expected = published_md5(session, name)
    if expected:
        actual = file_md5(zip_path)
        if actual != expected:
            zip_path.unlink()
            raise SystemExit(f"MD5 mismatch for {name}.zip ({actual} != {expected}); deleted it, run again.")
        log(f"MD5 verified ({actual})")
    else:
        log("Couldn't fetch the published MD5; skipping the checksum.")
    log(f"Extracting to {target.relative_to(ROOT)}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DATA)  # the zip holds a top-level <name>/ folder
    if not keep_zip:
        zip_path.unlink()
    sizes = {p.name: f"{p.stat().st_size / 1e6:.0f} MB" for p in sorted(target.glob("*.csv"))}
    log(f"Done: {sizes}")
    return target


def most_rated(dataset_dir: Path, count: int) -> list[tuple[int, str, str]]:
    """[(movieId, imdb 'tt…', title)] for the most-rated films, reading ratings in chunks."""
    import pandas as pd

    counts = None
    for chunk in pd.read_csv(dataset_dir / "ratings.csv", usecols=["movieId"], dtype={"movieId": "int32"},
                             chunksize=5_000_000):
        part = chunk["movieId"].value_counts()
        counts = part if counts is None else counts.add(part, fill_value=0)
    links = pd.read_csv(dataset_dir / "links.csv", dtype={"imdbId": str}).dropna(subset=["imdbId"])
    titles = pd.read_csv(dataset_dir / "movies.csv").set_index("movieId")["title"]
    links = links.assign(n=links["movieId"].map(counts).fillna(0)).sort_values("n", ascending=False).head(count)
    return [(int(r.movieId), f"tt{int(r.imdbId):07d}", str(titles.get(r.movieId, ""))) for r in links.itertuples()]


def fetch_posters(dataset_dir: Path, count: int, sleep: float) -> None:
    import wikipedia_source as wiki

    POSTERS.mkdir(parents=True, exist_ok=True)
    manifest_path = POSTERS / "manifest.csv"
    fields = ["movieId", "imdbId", "title", "article", "image_url", "file", "bytes", "status"]
    done: dict[int, dict] = {}
    if manifest_path.is_file():
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            done = {int(row["movieId"]): row for row in csv.DictReader(handle)}
    films = [f for f in most_rated(dataset_dir, count) if f[0] not in done]
    log(f"Posters: {len(done)} already handled, {len(films)} to fetch.")
    session = wiki._session()
    saved = 0
    new_file = not manifest_path.is_file() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for start in range(0, len(films), 50):
            batch = films[start:start + 50]
            try:
                # two API requests per 50 films: Wikidata for the article titles, Wikipedia for their lead images
                facts = wiki.wikidata_facts(session, [imdb for _m, imdb, _t in batch])
                titles = sorted({entry["article"] for entry in facts.values() if entry["article"]})
                images = wiki.page_images(session, titles) if titles else {}
            except (requests.RequestException, RuntimeError) as exc:
                log(f"  lookup failed ({exc}); retrying this batch next run.")
                continue
            for movie_id, imdb, title in batch:
                row = {"movieId": movie_id, "imdbId": imdb, "title": title, "article": "", "image_url": "",
                       "file": "", "bytes": 0, "status": ""}
                entry = facts.get(imdb)
                try:
                    image = images.get(entry["article"]) if entry else None
                    row["article"] = entry["article"] if entry else ""
                    if not image:
                        row["status"] = "no article" if not entry else "no image"
                    else:
                        response = wiki._request(session, image, timeout=30)
                        response.raise_for_status()
                        extension = Path(image.split("?")[0]).suffix.lower() or ".jpg"
                        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                            extension = ".jpg"
                        path = POSTERS / f"{movie_id}{extension}"
                        path.write_bytes(response.content)
                        row.update(image_url=image, file=path.name, bytes=len(response.content), status="ok")
                        saved += 1
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    row["status"] = f"error: {exc}"[:200]
                writer.writerow(row)
                handle.flush()
                time.sleep(sleep)
            log(f"  {min(start + 50, len(films))}/{len(films)} checked, {saved} posters saved")
    log(f"Posters done: {saved} new files in {POSTERS.relative_to(ROOT)}")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ml-25m")
    parser.add_argument("--posters", type=int, default=0, help="Also fetch posters for the N most-rated films.")
    parser.add_argument("--posters-only", action="store_true", help="Skip the dataset step (it must exist).")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip after extracting.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between poster downloads.")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    if args.dataset == "ml-latest-small":
        dataset_dir = DATA
    elif args.posters_only:
        dataset_dir = DATA / args.dataset
    else:
        dataset_dir = fetch_dataset(args.dataset, args.keep_zip)
    if args.posters:
        fetch_posters(dataset_dir, args.posters, args.sleep)


if __name__ == "__main__":
    main()
