"""Download the DBLP XML dump once and freeze it as a dated snapshot.

DBLP replaces dblp.xml.gz every few days, so the numbers in the report would change if the
notebook downloaded it again. This script is run once, outside the notebook, and everything
else reads the frozen copy.

Usage (from the project root):
    python scripts/download_dblp.py --date 2026-11-22

Output folder: data/raw/dblp_<date>/
    dblp.xml.gz       the dump
    dblp.dtd          the DTD that defines the character entities used in the dump
    dblp.xml.gz.md5   checksum published by DBLP
    metadata.json     source, retrieval time, server timestamp, size and checksum

An existing snapshot folder is never overwritten: if metadata.json is already there the
script stops and leaves everything as it is.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://dblp.org/xml/"
FILES = ("dblp.dtd", "dblp.xml.gz.md5")
USER_AGENT = "dblp-snapshot-downloader"
CHUNK = 1 << 20


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def fetch_small(url: str, target: Path) -> None:
    with urllib.request.urlopen(request(url)) as response:
        target.write_bytes(response.read())


def download_dump(url: str, target: Path) -> dict:
    part = target.with_suffix(target.suffix + ".part")
    digest = hashlib.md5()
    done = 0
    with urllib.request.urlopen(request(url)) as response, part.open("wb") as out:
        headers = {key.lower(): value for key, value in response.headers.items()}
        total = int(headers.get("content-length", 0))
        next_report = 0
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
                next_report += 100_000_000
    if total and done != total:
        sys.exit(f"Download incomplete: {done} of {total} bytes, kept {part}")
    part.replace(target)
    return {"md5": digest.hexdigest(), "size_bytes": done, "headers": headers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="snapshot date label, YYYY-MM-DD")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    folder = root / "data" / "raw" / f"dblp_{args.date}"
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        print(f"Snapshot {args.date} already exists, nothing to do: {folder}")
        return
    folder.mkdir(parents=True, exist_ok=True)

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Downloading dblp.xml.gz to {folder}")
    dump = download_dump(BASE_URL + "dblp.xml.gz", folder / "dblp.xml.gz")
    for name in FILES:
        print(f"Downloading {name}")
        fetch_small(BASE_URL + name, folder / name)

    server_md5 = (folder / "dblp.xml.gz.md5").read_text(encoding="utf-8").split()[0].lower()
    if server_md5 != dump["md5"]:
        sys.exit(f"Checksum mismatch: DBLP published {server_md5}, downloaded file is {dump['md5']}. Run again.")

    metadata = {
        "source": BASE_URL + "dblp.xml.gz",
        "dtd": BASE_URL + "dblp.dtd",
        "snapshot_date": args.date,
        "retrieved_at": retrieved_at,
        "last_modified": dump["headers"].get("last-modified", ""),
        "size_bytes": dump["size_bytes"],
        "md5": dump["md5"],
        "md5_matches_dblp": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Done:", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
