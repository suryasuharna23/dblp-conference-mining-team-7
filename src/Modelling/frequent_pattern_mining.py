"""Task 4 - Frequent Pattern Mining.

Apriori and FP-Growth are both implemented here from scratch on the same input, a
`TransactionDB`, and both return the same structure: {itemset: support count}. That makes
their results directly comparable in Task 6.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

MIN_SUPPORTS = (0.001, 0.005, 0.01, 0.02)

BAR_COLOR = "#3b6ea5"

Itemset = tuple[int, ...]


# --------------------------------------------------------------------------------------
# Transaction database
# --------------------------------------------------------------------------------------
@dataclass
class TransactionDB:
    """Transactions with venues encoded as integers.

    Many authors share exactly the same set of venues, so identical transactions are stored
    once with the number of times they occur. Support counts stay exact while both
    algorithms scan far fewer rows.
    """

    weighted: dict[Itemset, int]  # distinct transaction (sorted item ids) -> occurrences
    n_transactions: int
    items: list[str]  # item id -> venue name

    def min_count(self, min_support: float) -> int:
        """Smallest number of transactions an itemset needs to reach `min_support`."""
        return max(1, math.ceil(min_support * self.n_transactions - 1e-9))


def encode_transactions(item_tuples: pd.Series) -> TransactionDB:
    """Build a TransactionDB from a column of venue tuples."""
    occurrences = item_tuples.value_counts()
    items = sorted({item for transaction in occurrences.index for item in transaction})
    item_id = {item: i for i, item in enumerate(items)}
    weighted = {
        tuple(sorted(item_id[item] for item in transaction)): int(count)
        for transaction, count in occurrences.items()
    }
    return TransactionDB(weighted, int(occurrences.sum()), items)


# --------------------------------------------------------------------------------------
# Apriori
# --------------------------------------------------------------------------------------
def _generate_candidates(frequent: set[Itemset], k: int) -> set[Itemset]:
    """Apriori candidate generation: join frequent (k-1)-itemsets, then prune.

    Join: two frequent (k-1)-itemsets that share their first k-2 items form one k-candidate.
    Prune: a candidate is dropped when any of its (k-1)-subsets is not frequent, because a
    superset of an infrequent itemset can never be frequent (the Apriori property).
    """
    by_prefix: dict[Itemset, list[int]] = defaultdict(list)
    for itemset in frequent:
        by_prefix[itemset[:-1]].append(itemset[-1])

    candidates = set()
    for prefix, last_items in by_prefix.items():
        last_items.sort()
        for a, b in combinations(last_items, 2):
            candidate = prefix + (a, b)
            if all(subset in frequent for subset in combinations(candidate, k - 1)):
                candidates.add(candidate)
    return candidates


def _count_candidates(transactions: dict[Itemset, int], candidates: set[Itemset], k: int) -> dict[Itemset, int]:
    """One database scan that counts how many transactions contain each candidate."""
    counts = dict.fromkeys(candidates, 0)
    for transaction, weight in transactions.items():
        if math.comb(len(transaction), k) <= len(candidates):
            # Short transaction: list its k-subsets and look them up.
            for subset in combinations(transaction, k):
                if subset in counts:
                    counts[subset] += weight
        else:
            # Long transaction: it has more k-subsets than there are candidates.
            items = set(transaction)
            for candidate in candidates:
                if items.issuperset(candidate):
                    counts[candidate] += weight
    return counts


def _reduce_transactions(transactions: dict[Itemset, int], keep: set[int], min_length: int) -> dict[Itemset, int]:
    """Drop items that can no longer be part of a frequent itemset and merge equal transactions."""
    reduced: Counter = Counter()
    for transaction, weight in transactions.items():
        kept = tuple(item for item in transaction if item in keep)
        if len(kept) >= min_length:
            reduced[kept] += weight
    return reduced


def apriori(db: TransactionDB, min_support: float) -> dict[Itemset, int]:
    """Level-wise Apriori. Returns {itemset (sorted item ids): support count}."""
    min_count = db.min_count(min_support)

    item_counts: Counter = Counter()
    for transaction, weight in db.weighted.items():
        for item in transaction:
            item_counts[item] += weight
    frequent = {(item,): count for item, count in item_counts.items() if count >= min_count}
    result = dict(frequent)

    transactions = db.weighted
    k = 2
    while frequent:
        candidates = _generate_candidates(set(frequent), k)
        if not candidates:
            break
        # A transaction only matters at level k if it still holds k items used by a candidate.
        candidate_items = {item for candidate in candidates for item in candidate}
        transactions = _reduce_transactions(transactions, candidate_items, k)
        counts = _count_candidates(transactions, candidates, k)
        frequent = {itemset: count for itemset, count in counts.items() if count >= min_count}
        result.update(frequent)
        k += 1
    return result


# --------------------------------------------------------------------------------------
# FP-Growth
# --------------------------------------------------------------------------------------
class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[int, _Node] = {}
        self.link = None  # next node in the tree that holds the same item


def _build_tree(transactions, min_count: int) -> tuple[dict[int, _Node], dict[int, int]]:
    """Build an FP-tree from (items, weight) pairs.

    Returns (header, item_counts): header maps each frequent item to the first node of its
    node-link chain. Items are inserted from most to least frequent so that transactions
    share as many prefix nodes as possible, which keeps the tree small.
    """
    transactions = list(transactions)
    item_counts: Counter = Counter()
    for items, weight in transactions:
        for item in items:
            item_counts[item] += weight
    item_counts = {item: count for item, count in item_counts.items() if count >= min_count}

    root = _Node(None, None)
    header: dict[int, _Node] = {}
    for items, weight in transactions:
        ordered = sorted((item for item in items if item in item_counts), key=lambda item: (-item_counts[item], item))
        node = root
        for item in ordered:
            child = node.children.get(item)
            if child is None:
                child = _Node(item, node)
                node.children[item] = child
                child.link = header.get(item)
                header[item] = child
            child.count += weight
            node = child
    return header, item_counts


def _mine_tree(header, item_counts, suffix: Itemset, min_count: int, result: dict[Itemset, int]) -> None:
    """Recursive pattern growth: every frequent item extends the suffix, then its
    conditional FP-tree (built from the prefix paths that lead to it) is mined the same way."""
    for item in sorted(item_counts, key=lambda item: (item_counts[item], item)):
        pattern = suffix + (item,)
        result[tuple(sorted(pattern))] = item_counts[item]

        conditional_base = []
        node = header[item]
        while node is not None:
            path = []
            parent = node.parent
            while parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                conditional_base.append((path, node.count))
            node = node.link

        if conditional_base:
            cond_header, cond_counts = _build_tree(conditional_base, min_count)
            if cond_counts:
                _mine_tree(cond_header, cond_counts, pattern, min_count, result)


def fpgrowth(db: TransactionDB, min_support: float) -> dict[Itemset, int]:
    """FP-Growth. Returns {itemset (sorted item ids): support count}.

    The database is scanned twice (count the items, build the FP-tree) and no candidate
    itemsets are generated: patterns are grown directly from the tree.
    """
    min_count = db.min_count(min_support)
    header, item_counts = _build_tree(db.weighted.items(), min_count)
    result: dict[Itemset, int] = {}
    _mine_tree(header, item_counts, (), min_count, result)
    return result


ALGORITHMS = {"Apriori": apriori, "FP-Growth": fpgrowth}


# --------------------------------------------------------------------------------------
# Frequent itemsets as tables
# --------------------------------------------------------------------------------------
def itemsets_to_frame(itemsets: dict[Itemset, int], db: TransactionDB) -> pd.DataFrame:
    """Readable table: itemset (tuple of venue names), length, support_count, support."""
    frame = pd.DataFrame(
        {
            "itemset": [tuple(sorted(db.items[item] for item in itemset)) for itemset in itemsets],
            "length": [len(itemset) for itemset in itemsets],
            "support_count": list(itemsets.values()),
        }
    )
    frame["support"] = frame["support_count"] / db.n_transactions
    return frame.sort_values(["support_count", "itemset"], ascending=[False, True]).reset_index(drop=True)


def itemsets_of_length(itemsets: pd.DataFrame, length: int, n: int | None = None) -> pd.DataFrame:
    """Frequent k-itemsets, highest support first."""
    selected = itemsets[itemsets["length"] == length].reset_index(drop=True)
    return selected if n is None else selected.head(n)


def itemsets_by_length(itemsets: pd.DataFrame, lengths=(1, 2, 3), n: int = 10) -> pd.DataFrame:
    """The top n frequent k-itemsets for every k in `lengths`, side by side.

    Cells stay empty when fewer than n itemsets of that length are frequent.
    """
    columns = {}
    for length in lengths:
        selected = itemsets_of_length(itemsets, length, n)
        columns[f"{length}-itemsets"] = selected["itemset"].str.join(" + ")
        columns[f"support {length}"] = selected["support"].map("{:.3%}".format)
    return pd.DataFrame(columns).reindex(range(n)).dropna(how="all").fillna("")


def top_itemsets(itemsets: pd.DataFrame, n: int = 10, min_length: int = 1) -> pd.DataFrame:
    """The n itemsets with the highest support. Use min_length=2 for venue combinations."""
    return itemsets[itemsets["length"] >= min_length].head(n).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------
@dataclass
class MiningResult:
    algorithm: str
    min_support: float
    n_transactions: int
    itemsets: pd.DataFrame
    runtime_s: float  # median time spent inside the mining algorithm only


def mine(db: TransactionDB, algorithm: str, min_support: float, repeats: int = 1) -> MiningResult:
    """Run one algorithm at one minimum support.

    The algorithm is deterministic, so repeating it only serves the runtime measurement:
    the median of `repeats` runs is less sensitive to whatever else the machine is doing.
    """
    runtimes = []
    for _ in range(repeats):
        start = time.perf_counter()
        raw_itemsets = ALGORITHMS[algorithm](db, min_support)
        runtimes.append(time.perf_counter() - start)
    itemsets = itemsets_to_frame(raw_itemsets, db)
    return MiningResult(algorithm, min_support, db.n_transactions, itemsets, statistics.median(runtimes))


def run_experiments(
    db: TransactionDB, min_supports=MIN_SUPPORTS, algorithms=tuple(ALGORITHMS), repeats: int = 1
) -> dict[tuple[str, float], MiningResult]:
    """Every algorithm at every minimum support. Keys are (algorithm, min_support)."""
    return {
        (algorithm, min_support): mine(db, algorithm, min_support, repeats)
        for min_support in min_supports
        for algorithm in algorithms
    }


def itemset_count_table(results: dict[tuple[str, float], MiningResult], algorithm: str) -> pd.DataFrame:
    """Number of frequent k-itemsets (columns) per minimum support (rows) for one algorithm."""
    rows = {}
    for (name, min_support), result in results.items():
        if name != algorithm:
            continue
        by_length = result.itemsets["length"].value_counts().sort_index()
        row = {f"{length}-itemsets": count for length, count in by_length.items()}
        row["total"] = len(result.itemsets)
        row["max_length"] = int(result.itemsets["length"].max()) if len(result.itemsets) else 0
        rows[f"{min_support:.1%}"] = row
    table = pd.DataFrame(rows).T.fillna(0).astype(int).rename_axis("min_support")
    ordered = sorted((c for c in table.columns if c.endswith("-itemsets")), key=lambda c: int(c.split("-")[0]))
    return table[ordered + ["total", "max_length"]]


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_itemset_counts(count_table: pd.DataFrame):
    """Stacked bars: number of frequent itemsets per minimum support, split by itemset length."""
    lengths = [column for column in count_table.columns if column.endswith("-itemsets")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.get_cmap("viridis", len(lengths))
    bottom = pd.Series(0, index=count_table.index)
    for i, column in enumerate(lengths):
        ax.bar(count_table.index, count_table[column], bottom=bottom, color=colors(i), label=column)
        bottom = bottom + count_table[column]
    for x, total in zip(count_table.index, count_table["total"]):
        ax.text(x, total, f"{total:,}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Frequent itemsets per minimum support", xlabel="Minimum support", ylabel="Frequent itemsets")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_top_itemsets(itemsets: pd.DataFrame, n: int = 10, min_length: int = 2, title: str | None = None):
    top = top_itemsets(itemsets, n, min_length).iloc[::-1]
    labels = top["itemset"].str.join(" + ")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top) + 1.5))
    ax.barh(labels, top["support"] * 100, color=BAR_COLOR)
    for y, (support, count) in enumerate(zip(top["support"], top["support_count"])):
        ax.text(support * 100, y, f" {support:.2%} ({count:,} authors)", va="center", fontsize=8)
    ax.set(title=title or f"Top {n} venue combinations by support", xlabel="Support (%)")
    ax.set_xlim(0, top["support"].max() * 100 * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
