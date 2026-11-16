"""Task 1 - Data Understanding and Preprocessing.

Pipeline covered here:
    download_dblp      -> fetch the official DBLP XML dump and record when it was retrieved
    extract_dblp       -> one streaming pass over the dump: count every record type and
                          keep all `inproceedings` records in a parquet file
    preprocess         -> select 2015-2025 conference/workshop papers, handle missing values,
                          normalise authors and venues, fix years, drop duplicates
    statistics / plots -> dataset statistics after preprocessing (Task 1.4)
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

DBLP_XML_URL = "https://dblp.org/xml/dblp.xml.gz"
DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

YEAR_MIN, YEAR_MAX = 2015, 2025

# Every record type defined in dblp.dtd.
RECORD_TAGS = (
    "article",
    "inproceedings",
    "proceedings",
    "book",
    "incollection",
    "phdthesis",
    "mastersthesis",
    "www",
    "person",
    "data",
)

BAR_COLOR = "#3b6ea5"
ACCENT_COLOR = "#d98032"


# --------------------------------------------------------------------------------------
# Data extraction
# --------------------------------------------------------------------------------------
def download_dblp(raw_dir: Path = RAW_DIR, force: bool = False) -> dict:
    """Download dblp.xml.gz and dblp.dtd, and write the retrieval metadata next to them.

    The dump on dblp.org is replaced regularly, so the retrieval date and the server's
    Last-Modified header are stored in `metadata.json` to make the snapshot identifiable.
    Existing files are reused unless `force` is set.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / "metadata.json"
    xml_path = raw_dir / "dblp.xml.gz"
    dtd_path = raw_dir / "dblp.dtd"

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for url, path in ((DBLP_DTD_URL, dtd_path), (DBLP_XML_URL, xml_path)):
        if path.exists() and not force:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        tmp_path = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
            if path == xml_path:
                meta["last_modified"] = response.headers.get("Last-Modified")
        tmp_path.replace(path)
        if path == xml_path:
            meta["retrieved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if "retrieved_at" not in meta:
        # File was placed manually: fall back to its modification time.
        mtime = datetime.fromtimestamp(xml_path.stat().st_mtime, timezone.utc)
        meta["retrieved_at"] = mtime.strftime("%Y-%m-%d %H:%M UTC")
    meta.update(
        source=DBLP_XML_URL,
        dtd=DBLP_DTD_URL,
        size_bytes=xml_path.stat().st_size,
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _text(element) -> str:
    """Full text of an element, including text inside nested tags such as <i> or <sub>."""
    return "".join(element.itertext()).strip()


def extract_dblp(
    raw_dir: Path = RAW_DIR, out_dir: Path = INTERIM_DIR, force: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Stream the DBLP dump once and return (raw inproceedings, dump summary).

    The dump is about 5 GB of XML, so it is parsed as a stream and every finished record
    is released from memory straight away. All record types are counted for the data
    understanding step, but only `inproceedings` records are stored, untouched, so that
    every cleaning decision stays visible in `preprocess`.

    The summary holds the number of records per publication type and, per type, how many
    records carry each attribute.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "inproceedings_raw.parquet"
    summary_path = out_dir / "dump_summary.json"
    if parquet_path.exists() and summary_path.exists() and not force:
        return pd.read_parquet(parquet_path), json.loads(summary_path.read_text(encoding="utf-8"))

    schema = pa.schema(
        [
            ("key", pa.string()),
            ("mdate", pa.string()),
            ("title", pa.string()),
            ("year", pa.string()),
            ("booktitle", pa.string()),
            ("authors", pa.list_(pa.string())),
        ]
    )
    type_counts: Counter = Counter()
    field_counts: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict] = []

    # The DTD declares the character entities (&uuml; etc.) used in the dump. base_url lets
    # the parser find dblp.dtd beside the dump even though the data arrives as a byte stream.
    parser = etree.XMLPullParser(
        events=("end",),
        tag=RECORD_TAGS,
        load_dtd=True,
        resolve_entities=True,
        no_network=True,
        huge_tree=True,
        base_url=str(raw_dir / "dblp.xml"),
    )
    tmp_path = parquet_path.with_suffix(".parquet.part")
    with gzip.open(raw_dir / "dblp.xml.gz", "rb") as stream, pq.ParquetWriter(tmp_path, schema) as writer:
        while chunk := stream.read(1 << 24):
            parser.feed(chunk)
            for _, element in parser.read_events():
                tag = element.tag
                type_counts[tag] += 1
                field_counts[tag].update({child.tag for child in element})
                if tag == "inproceedings":
                    rows.append(
                        {
                            "key": element.get("key"),
                            "mdate": element.get("mdate"),
                            "title": _text(element.find("title")) if element.find("title") is not None else None,
                            "year": element.findtext("year"),
                            "booktitle": element.findtext("booktitle"),
                            "authors": [_text(author) for author in element.findall("author")],
                        }
                    )
                # Free the finished record and everything parsed before it.
                element.clear()
                parent = element.getparent()
                while element.getprevious() is not None:
                    del parent[0]
            if len(rows) >= 200_000:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    tmp_path.replace(parquet_path)

    summary = {
        "record_types": dict(type_counts.most_common()),
        "fields_per_type": {tag: dict(counter.most_common()) for tag, counter in field_counts.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pd.read_parquet(parquet_path), summary


# --------------------------------------------------------------------------------------
# Task 1.1 - data understanding
# --------------------------------------------------------------------------------------
def record_type_table(summary: dict) -> pd.DataFrame:
    """Number of records per publication type in the full dump."""
    table = pd.Series(summary["record_types"], name="records").rename_axis("publication_type").to_frame()
    table["share_%"] = (table["records"] / table["records"].sum() * 100).round(2)
    return table


def attribute_table(summary: dict, record_type: str = "inproceedings") -> pd.DataFrame:
    """Attributes available for one record type and how many records carry each of them."""
    total = summary["record_types"][record_type]
    table = pd.Series(summary["fields_per_type"][record_type], name="records").rename_axis("attribute").to_frame()
    table["coverage_%"] = (table["records"] / total * 100).round(2)
    return table


def missing_value_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Missing or empty values per extracted attribute of the raw inproceedings records."""
    missing = {
        "key": raw["key"].isna().sum(),
        "title": (raw["title"].isna() | (raw["title"].str.strip() == "")).sum(),
        "year": raw["year"].isna().sum(),
        "booktitle": (raw["booktitle"].isna() | (raw["booktitle"].str.strip() == "")).sum(),
        "authors": (raw["authors"].str.len() == 0).sum(),
    }
    table = pd.Series(missing, name="missing").rename_axis("attribute").to_frame()
    table["missing_%"] = (table["missing"] / len(raw) * 100).round(3)
    return table


# --------------------------------------------------------------------------------------
# Task 1.2 - preprocessing
# --------------------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")
_PART_MARKER = re.compile(r"\s*\((?:\d+(?:-\d+)?|[IVXLC]+)\)")
_YEAR_TOKEN = re.compile(r"\s*\b(?:19|20)\d{2}\b")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_author(name: str) -> str:
    """Unicode-normalise an author name and collapse its whitespace.

    DBLP already disambiguates people: a homonym suffix such as "Wei Wang 0001" marks one
    specific person. That suffix is kept on purpose, because removing it would merge
    thousands of different researchers into one transaction.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", name)).strip()


def clean_booktitle(booktitle: str) -> str:
    """Remove the parts of a booktitle that describe one edition instead of the venue.

    "ECCV (17)" -> "ECCV", "ICML 2019" -> "ICML". Volume markers and years would otherwise
    split a single conference into many different items.
    """
    text = unicodedata.normalize("NFKC", booktitle)
    text = _PART_MARKER.sub("", text)
    text = _YEAR_TOKEN.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip(" -,:/")
    return text or _WHITESPACE.sub(" ", booktitle).strip()


def normalize_venues(papers: pd.DataFrame) -> pd.DataFrame:
    """Add `stream` and the canonical `venue` name to every paper.

    The venue identity is the DBLP stream in the record key (`conf/nips/...` -> `conf/nips`).
    DBLP curates that stream by hand, so it stays the same when a conference is renamed
    (NIPS -> NeurIPS), written differently (INTERSPEECH / Interspeech), split into volumes
    ("ECCV (17)") or tracks ("ACL (Findings)"). Workshops that DBLP files under their host
    conference ("CVPR Workshops", "BlackboxNLP@EMNLP") therefore count as that conference,
    while workshops with a stream of their own stay separate venues.

    A few conference papers were published in a journal series and carry a `journals/` or
    `series/` key that is shared by unrelated conferences. For those the cleaned booktitle
    is the venue identity. The readable venue name is the most common cleaned booktitle of
    the stream.
    """
    papers = papers.copy()
    cleaned = papers["booktitle"].map({b: clean_booktitle(b) for b in papers["booktitle"].unique()})
    papers["stream"] = papers["key"].str.split("/").str[:2].str.join("/")
    in_conf_stream = papers["key"].str.startswith("conf/")
    papers.loc[~in_conf_stream, "stream"] = "booktitle/" + cleaned[~in_conf_stream].str.casefold()

    counts = (
        pd.DataFrame({"stream": papers["stream"], "name": cleaned})
        .value_counts()
        .rename("n")
        .reset_index()
        .sort_values(["stream", "n", "name"], ascending=[True, False, True])
    )
    names = counts.drop_duplicates("stream").set_index("stream")["name"]
    # Different streams can share a name (four conferences are called "ICIT"). The largest
    # one keeps the plain name, the others get their stream id so every venue stays distinct.
    sizes = papers["stream"].value_counts()
    names = names.loc[sizes.index]
    clashes = names[names.duplicated(keep="first")]
    names.loc[clashes.index] = [f"{name} [{stream}]" for stream, name in clashes.items()]
    papers["venue"] = papers["stream"].map(names)
    return papers


def preprocess(
    raw: pd.DataFrame, n_dump_records: int | None = None, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean the raw inproceedings records.

    `n_dump_records` is the number of records of all types in the dump; when given, the log
    also shows how many records the selection of conference/workshop papers removed.

    Returns (papers, authorship, log):
        papers      one row per publication: key, title, year, venue, stream, booktitle, n_authors
        authorship  one row per (publication, author): key, author, year, venue
        log         rows removed by every preprocessing step
    """
    log: list[dict] = []

    def record(step: str, before: int, frame: pd.DataFrame, note: str) -> None:
        log.append({"step": step, "rows_before": before, "rows_after": len(frame), "removed": before - len(frame), "note": note})

    # 1. Conference and workshop publications only: DBLP stores them as `inproceedings`.
    df = raw.copy()
    record("select conference/workshop", n_dump_records or len(raw), df, "kept record type inproceedings")

    # 2. Year consistency: a numeric four digit year inside the observation period.
    before = len(df)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(year_min, year_max)].copy()
    df["year"] = df["year"].astype("int16")
    record("year consistency", before, df, f"year cast to integer, kept {year_min}-{year_max}")

    # 3. Missing values: author, venue and title are required to build a transaction.
    before = len(df)
    df["title"] = df["title"].fillna("").str.strip()
    df["booktitle"] = df["booktitle"].fillna("").str.strip()
    df = df[(df["authors"].str.len() > 0) & (df["booktitle"] != "") & (df["title"] != "")].copy()
    record("missing values", before, df, "dropped records without author, booktitle or title")

    # 4. Venue normalisation.
    df = normalize_venues(df)
    record("venue normalisation", len(df), df, f"{df['booktitle'].nunique():,} booktitles -> {df['venue'].nunique():,} venues")

    # 5. Author normalisation, then one row per (paper, author).
    authorship = df[["key", "authors"]].explode("authors").rename(columns={"authors": "author"})
    names_before = authorship["author"].nunique()
    authorship["author"] = authorship["author"].map({a: normalize_author(a) for a in authorship["author"].unique()})
    authorship = authorship[authorship["author"] != ""].drop_duplicates(["key", "author"])
    record("author normalisation", len(df), df, f"{names_before:,} raw names -> {authorship['author'].nunique():,} authors")

    # 6. Duplicates: same key, or the same paper entered twice in one venue and year.
    before = len(df)
    df = df.drop_duplicates("key")
    author_sets = authorship.sort_values("author").groupby("key", sort=False)["author"].agg("|".join)
    df["_title"] = df["title"].str.casefold().str.replace(_NON_ALNUM, "", regex=True)
    df["_authors"] = df["key"].map(author_sets)
    df = df.drop_duplicates(["_title", "_authors", "venue", "year"]).drop(columns=["_title", "_authors"])
    record("duplicates", before, df, "same normalised title, author set, venue and year")

    authorship = authorship[authorship["key"].isin(df["key"])]
    df["n_authors"] = df["key"].map(authorship.groupby("key").size()).astype("int16")
    papers = df[["key", "title", "year", "venue", "stream", "booktitle", "n_authors"]].reset_index(drop=True)
    authorship = authorship.merge(papers[["key", "year", "venue"]], on="key").reset_index(drop=True)
    return papers, authorship, pd.DataFrame(log)


def save_processed(papers: pd.DataFrame, authorship: pd.DataFrame, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    papers.to_parquet(out_dir / "papers.parquet", index=False)
    authorship.to_parquet(out_dir / "authorship.parquet", index=False)


# --------------------------------------------------------------------------------------
# Task 1.4 - statistics after preprocessing
# --------------------------------------------------------------------------------------
def dataset_statistics(papers: pd.DataFrame, authorship: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "total publications": len(papers),
            "unique authors": authorship["author"].nunique(),
            "unique venues": papers["venue"].nunique(),
            "author-paper pairs": len(authorship),
            "mean authors per paper": round(papers["n_authors"].mean(), 2),
            "first year": papers["year"].min(),
            "last year": papers["year"].max(),
        },
        name="value",
        dtype=object,
    )


def publications_per_year(papers: pd.DataFrame) -> pd.Series:
    return papers["year"].value_counts().sort_index().rename("publications")


def publications_per_venue(papers: pd.DataFrame) -> pd.Series:
    return papers["venue"].value_counts().rename("publications")


def plot_publications_per_year(papers: pd.DataFrame):
    per_year = publications_per_year(papers)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(per_year.index.astype(str), per_year.values, color=BAR_COLOR)
    for x, y in zip(per_year.index.astype(str), per_year.values):
        ax.text(x, y, f"{y / 1000:.0f}k", ha="center", va="bottom", fontsize=8)
    ax.set(title="Publications per year", xlabel="Year", ylabel="Publications")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_venue_size_distribution(papers: pd.DataFrame):
    """Venue sizes are heavily skewed, so the histogram uses a log scale on the x axis."""
    sizes = publications_per_venue(papers)
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = [10 ** (i / 4) for i in range(0, 4 * 5 + 2)]
    ax.hist(sizes.values, bins=bins, color=BAR_COLOR, edgecolor="white")
    ax.set_xscale("log")
    ax.set(title="Distribution of publications per venue", xlabel="Publications per venue (log scale)", ylabel="Venues")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
