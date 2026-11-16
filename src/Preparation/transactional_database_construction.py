"""Task 3 - Transactional Database Construction.

Two transactional databases are built from the cleaned author-paper table:
    author-based       one transaction per author, items = venues the author published in
                       at any time in the observation period
    author-year-based  one transaction per (author, year), items = venues the author
                       published in during that year

A transaction is a set, so a venue appears once no matter how many papers the author has there.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PERIODS = {
    "2015-2017": (2015, 2017),
    "2018-2020": (2018, 2020),
    "2021-2023": (2021, 2023),
    "2024-2025": (2024, 2025),
}

COLORS = {"author": "#3b6ea5", "author-year": "#d98032"}


def _collect(authorship: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    unique = authorship[by + ["venue"]].drop_duplicates().sort_values(by + ["venue"])
    transactions = unique.groupby(by, sort=False)["venue"].agg(tuple).rename("items").reset_index()
    transactions["n_items"] = transactions["items"].str.len()
    return transactions


def build_author_transactions(authorship: pd.DataFrame) -> pd.DataFrame:
    """One row per author: author, items (sorted tuple of venues), n_items."""
    return _collect(authorship, ["author"])


def build_author_year_transactions(authorship: pd.DataFrame) -> pd.DataFrame:
    """One row per (author, year): author, year, items (sorted tuple of venues), n_items."""
    return _collect(authorship, ["author", "year"])


def split_by_period(author_year_tx: pd.DataFrame, periods: dict = PERIODS) -> dict[str, pd.DataFrame]:
    """Author-year transactions of every period, used by the temporal association analysis."""
    return {
        name: author_year_tx[author_year_tx["year"].between(start, end)].reset_index(drop=True)
        for name, (start, end) in periods.items()
    }


def sample_transactions(transactions: pd.DataFrame, n: int = 10, min_items: int = 2, seed: int = 42) -> pd.DataFrame:
    """A reproducible sample of transactions that hold at least `min_items` items.

    Most transactions hold a single venue, so the sample is taken from the multi-item ones
    to show what a co-occurrence looks like.
    """
    return transactions[transactions["n_items"] >= min_items].sample(n, random_state=seed).reset_index(drop=True)


def transaction_statistics(transactions: pd.DataFrame) -> pd.Series:
    """Number of transactions, unique items and a summary of the transaction length."""
    sizes = transactions["n_items"]
    return pd.Series(
        {
            "transactions": len(transactions),
            "unique items": int(transactions["items"].explode().nunique()),
            "total items": int(sizes.sum()),
            "mean items per transaction": round(sizes.mean(), 3),
            "median items per transaction": int(sizes.median()),
            "max items per transaction": int(sizes.max()),
            "transactions with 1 item (%)": round((sizes == 1).mean() * 100, 2),
            "transactions with >= 2 items (%)": round((sizes >= 2).mean() * 100, 2),
            "transactions with >= 3 items (%)": round((sizes >= 3).mean() * 100, 2),
        },
        dtype=object,
    )


def compare_representations(author_tx: pd.DataFrame, author_year_tx: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "author-based": transaction_statistics(author_tx),
            "author-year-based": transaction_statistics(author_year_tx),
        }
    )


def size_distribution(transactions: pd.DataFrame, cap: int = 10) -> pd.DataFrame:
    """Number and share of transactions per transaction length; lengths >= cap share one row."""
    capped = transactions["n_items"].clip(upper=cap)
    counts = capped.value_counts().sort_index()
    table = pd.DataFrame({"transactions": counts, "share_%": (counts / len(transactions) * 100).round(2)})
    table.index = [f">={cap}" if size == cap else str(size) for size in table.index]
    return table.rename_axis("items_per_transaction")


def plot_size_distribution(author_tx: pd.DataFrame, author_year_tx: pd.DataFrame, cap: int = 10):
    """Share of transactions per transaction length for both representations, side by side."""
    shares = pd.DataFrame(
        {
            "author": size_distribution(author_tx, cap)["share_%"],
            "author-year": size_distribution(author_year_tx, cap)["share_%"],
        }
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = range(len(shares))
    width = 0.4
    ax.bar([p - width / 2 for p in positions], shares["author"], width, color=COLORS["author"], label="Author-based")
    ax.bar([p + width / 2 for p in positions], shares["author-year"], width, color=COLORS["author-year"], label="Author-year-based")
    ax.set_xticks(list(positions), shares.index)
    ax.set_yscale("log")
    ax.set(title="Items per transaction", xlabel="Items per transaction", ylabel="Share of transactions (%, log scale)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def save_transactions(author_tx: pd.DataFrame, author_year_tx: pd.DataFrame, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    author_tx.assign(items=author_tx["items"].map(list)).to_parquet(out_dir / "transactions_author.parquet", index=False)
    author_year_tx.assign(items=author_year_tx["items"].map(list)).to_parquet(
        out_dir / "transactions_author_year.parquet", index=False
    )
