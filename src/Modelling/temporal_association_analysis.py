"""Section E - Temporal Association Analysis.

The author-year transactions are split into periods, every period is mined separately and
the frequent itemsets and association rules are compared between periods.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from Modelling.frequent_pattern_mining import MiningResult, encode_transactions, mine
from Modelling.association_rule_mining import MIN_CONFIDENCE, association_rules

BAR_COLOR = "#3b6ea5"


def mine_periods(
    period_transactions: dict[str, pd.DataFrame],
    min_support: float,
    min_confidence: float = MIN_CONFIDENCE,
    algorithm: str = "FP-Growth",
) -> tuple[dict[str, MiningResult], dict[str, pd.DataFrame]]:
    """Mine every period separately. Returns (frequent itemset results, rules), both keyed by period."""
    results, rules = {}, {}
    for period, transactions in period_transactions.items():
        result = mine(encode_transactions(transactions["items"]), algorithm, min_support)
        results[period] = result
        rules[period] = association_rules(result.itemsets, result.n_transactions, min_confidence)
    return results, rules


def period_summary(period_results: dict, period_rules: dict) -> pd.DataFrame:
    """Transactions, frequent itemsets per length and rules of every period."""
    rows = {}
    for period, result in period_results.items():
        by_length = result.itemsets["length"].value_counts()
        rows[period] = {
            "transactions": result.n_transactions,
            "frequent_itemsets": len(result.itemsets),
            "1-itemsets": int(by_length.get(1, 0)),
            "2-itemsets": int(by_length.get(2, 0)),
            "3+-itemsets": int(by_length[by_length.index >= 3].sum()),
            "association_rules": len(period_rules[period]),
        }
    return pd.DataFrame(rows).T.rename_axis("period")


def _add_pattern(table: pd.DataFrame) -> pd.DataFrame:
    """Add `periods_present` and `pattern` to a table with one column per period (NaN = absent).

        persistent  found in every period
        emerging    absent in the first period, present in the last
        vanished    present in the first period, absent in the last
        temporary   anything else (appears and disappears in between)
    """
    periods = list(table.columns)
    present = table.notna()
    table["periods_present"] = present.sum(axis=1)
    table["pattern"] = "temporary"
    table.loc[present[periods[0]] & ~present[periods[-1]], "pattern"] = "vanished"
    table.loc[~present[periods[0]] & present[periods[-1]], "pattern"] = "emerging"
    table.loc[present.all(axis=1), "pattern"] = "persistent"
    return table


def _rule_text(rules: pd.DataFrame) -> pd.Series:
    return rules["antecedent"].str.join(", ") + "  ->  " + rules["consequent"].str.join(", ")


def rule_persistence(period_rules: dict, measure: str = "lift") -> pd.DataFrame:
    """One row per rule, one column per period holding the rule's `measure` (NaN = rule absent).

    `pattern` describes how the rule behaves over time, see `_add_pattern`.
    """
    periods = list(period_rules)
    columns = {}
    for period, rules in period_rules.items():
        columns[period] = pd.Series(rules[measure].values, index=_rule_text(rules), dtype=float)
    table = _add_pattern(pd.DataFrame(columns).reindex(columns=periods))
    return table.sort_values(["periods_present", periods[-1]], ascending=False).rename_axis("rule")


def pattern_counts(persistence: pd.DataFrame) -> pd.Series:
    return persistence["pattern"].value_counts().rename("rules")


def itemset_support_over_time(period_results: dict, min_length: int = 2) -> pd.DataFrame:
    """Support (%) of every frequent venue combination per period (NaN = not frequent),
    with the same `pattern` labels as `rule_persistence`."""
    columns = {}
    for period, result in period_results.items():
        selected = result.itemsets[result.itemsets["length"] >= min_length]
        columns[period] = pd.Series(
            (selected["support"] * 100).values, index=selected["itemset"].str.join(" + "), dtype=float
        )
    periods = list(period_results)
    table = _add_pattern(pd.DataFrame(columns).reindex(columns=periods))
    return table.sort_values(["periods_present", periods[-1]], ascending=False).rename_axis("itemset")


def plot_period_summary(period_results: dict, period_rules: dict):
    summary = period_summary(period_results, period_rules)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("transactions", "Transactions"), ("frequent_itemsets", "Frequent itemsets"), ("association_rules", "Association rules")]
    for ax, (column, title) in zip(axes, panels):
        ax.bar(summary.index, summary[column], color=BAR_COLOR)
        for x, value in zip(summary.index, summary[column]):
            ax.text(x, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
        ax.set(title=f"{title} per period")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_period_heatmap(persistence: pd.DataFrame, n: int = 25, measure: str = "lift", title: str | None = None):
    """Heatmap of the first n rows of a `rule_persistence` or `itemset_support_over_time`
    table; blank cells mean the rule or itemset is absent in that period."""
    periods = [column for column in persistence.columns if column not in ("periods_present", "pattern")]
    table = persistence.head(n)[periods]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(table) + 1.5))
    image = ax.imshow(table.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(periods)), periods)
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    middle = (table.min().min() + table.max().max()) / 2
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iat[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < middle else "black")
    fig.colorbar(image, ax=ax, label=measure)
    ax.set_title(title or f"{measure} per period")
    fig.tight_layout()
    return fig
