"""Task 5 - Association Rule Mining.

Rules are generated from the frequent itemsets found in Task 4 and measured with support,
confidence and lift.
"""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

MIN_CONFIDENCE = 0.5

BAR_COLOR = "#3b6ea5"


# --------------------------------------------------------------------------------------
# Association rules
# --------------------------------------------------------------------------------------
RULE_COLUMNS = [
    "antecedent",
    "consequent",
    "support",
    "confidence",
    "lift",
    "support_count",
    "antecedent_support",
    "consequent_support",
]


def association_rules(itemsets: pd.DataFrame, n_transactions: int, min_confidence: float = MIN_CONFIDENCE) -> pd.DataFrame:
    """Generate every rule A -> B with confidence >= min_confidence from frequent itemsets.

    For a frequent itemset I and a split into A and B = I - A:
        support(A -> B)    = count(I) / N              how often A and B occur together
        confidence(A -> B) = count(I) / count(A)       share of A's transactions that also hold B
        lift(A -> B)       = confidence / support(B)   how much more often than by chance
    Every subset of a frequent itemset is frequent too, so count(A) and count(B) are always
    available in the mined itemsets.
    """
    counts = dict(zip(itemsets["itemset"], itemsets["support_count"]))
    rows = []
    for itemset, count in counts.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                confidence = count / counts[antecedent]
                if confidence < min_confidence:
                    continue
                consequent = tuple(item for item in itemset if item not in antecedent)
                consequent_support = counts[consequent] / n_transactions
                rows.append(
                    (
                        antecedent,
                        consequent,
                        count / n_transactions,
                        confidence,
                        confidence / consequent_support,
                        count,
                        counts[antecedent] / n_transactions,
                        consequent_support,
                    )
                )
    rules = pd.DataFrame(rows, columns=RULE_COLUMNS)
    # Keep numeric dtypes even when no rule passes the confidence threshold.
    rules = rules.astype({column: float for column in RULE_COLUMNS[2:]}).astype({"support_count": int})
    return rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def top_rules(rules: pd.DataFrame, n: int = 10, by: str = "lift") -> pd.DataFrame:
    return rules.sort_values([by, "confidence", "support"], ascending=False).head(n).reset_index(drop=True)


def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Rules as text ("A, B -> C") with rounded measures, for display."""
    table = pd.DataFrame(
        {
            "rule": rules["antecedent"].str.join(", ") + "  ->  " + rules["consequent"].str.join(", "),
            "support": rules["support"].round(5),
            "confidence": rules["confidence"].round(3),
            "lift": rules["lift"].round(2),
            "support_count": rules["support_count"],
        }
    )
    return table.reset_index(drop=True)


def generate_rules(results: dict, min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Association rules of every Task 4 experiment. Keys match `results`: (algorithm, min_support)."""
    return {
        key: association_rules(result.itemsets, result.n_transactions, min_confidence)
        for key, result in results.items()
    }


def rule_count_table(rules: dict, algorithm: str) -> pd.DataFrame:
    """Number of rules and the range of their measures per minimum support for one algorithm."""
    rows = {}
    for (name, min_support), frame in rules.items():
        if name != algorithm:
            continue
        rows[f"{min_support:.1%}"] = {
            "rules": len(frame),
            "max_lift": frame["lift"].max(),
            "median_lift": frame["lift"].median(),
            "max_confidence": frame["confidence"].max(),
            "max_support": frame["support"].max(),
        }
    return pd.DataFrame(rows).T.rename_axis("min_support").astype({"rules": int})


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_rules(rules: pd.DataFrame, title: str = "Association rules"):
    """Support against confidence for every rule, coloured by lift."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    points = ax.scatter(
        rules["support"] * 100, rules["confidence"], c=rules["lift"], cmap="viridis", norm="log", s=28, alpha=0.8
    )
    fig.colorbar(points, ax=ax, label="Lift (log scale)")
    ax.set_xscale("log")
    ax.set(title=title, xlabel="Support (%, log scale)", ylabel="Confidence")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_support_vs_lift(rules: pd.DataFrame):
    """Task 5.4: does a high support come with a high lift?"""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(rules["support"] * 100, rules["lift"], color=BAR_COLOR, s=24, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    correlation = rules["support"].corr(rules["lift"], method="spearman")
    ax.set(
        title=f"Rule support against lift (Spearman correlation {correlation:.2f})",
        xlabel="Support (%, log scale)",
        ylabel="Lift (log scale)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
