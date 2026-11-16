"""Task 6 - Algorithm Performance Comparison.

Compares Apriori and FP-Growth on the same transactional database and the same minimum
supports. Inputs are the Task 4 results (`MiningResult` per experiment) and the Task 5
rules, both keyed by (algorithm, min_support).
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from scipy.sparse import csr_matrix

COLORS = {"Apriori": "#3b6ea5", "FP-Growth": "#d98032"}


def performance_table(results: dict, rules: dict) -> pd.DataFrame:
    """One row per (minimum support, algorithm): frequent itemsets, rules and runtime."""
    rows = [
        {
            "min_support": f"{min_support:.1%}",
            "algorithm": algorithm,
            "frequent_itemsets": len(result.itemsets),
            "association_rules": len(rules[(algorithm, min_support)]),
            "max_itemset_length": int(result.itemsets["length"].max()) if len(result.itemsets) else 0,
            "runtime_s": round(result.runtime_s, 3),
        }
        for (algorithm, min_support), result in results.items()
    ]
    return pd.DataFrame(rows).set_index(["min_support", "algorithm"])


def _itemset_counts(result) -> dict:
    return dict(zip(result.itemsets["itemset"], result.itemsets["support_count"]))


def _rule_set(rules: pd.DataFrame) -> set:
    return set(zip(rules["antecedent"], rules["consequent"]))


def compare_results(results: dict, rules: dict, first: str = "Apriori", second: str = "FP-Growth") -> pd.DataFrame:
    """Check per minimum support whether both algorithms found the same itemsets,
    the same support counts and the same rules."""
    rows = []
    for min_support in sorted({min_support for _, min_support in results}):
        a, b = results[(first, min_support)], results[(second, min_support)]
        counts_a, counts_b = _itemset_counts(a), _itemset_counts(b)
        rules_a = _rule_set(rules[(first, min_support)])
        rules_b = _rule_set(rules[(second, min_support)])
        rows.append(
            {
                "min_support": f"{min_support:.1%}",
                f"itemsets_{first}": len(counts_a),
                f"itemsets_{second}": len(counts_b),
                "itemsets_in_both": len(counts_a.keys() & counts_b.keys()),
                "same_itemsets": counts_a.keys() == counts_b.keys(),
                "same_support_counts": counts_a == counts_b,
                "same_rules": rules_a == rules_b,
            }
        )
    return pd.DataFrame(rows).set_index("min_support")


def crosscheck_with_mlxtend(db, result) -> pd.Series:
    """Test our implementation against the FP-Growth of the mlxtend library.

    Only the columns of frequent single items are one-hot encoded: an infrequent item can
    not be part of any frequent itemset, so leaving it out does not change the result and
    keeps the matrix small enough for memory.
    """
    min_count = db.min_count(result.min_support)
    frequent_items = result.itemsets.loc[result.itemsets["length"] == 1, "itemset"].str[0]
    column = {db.items.index(name): i for i, name in enumerate(frequent_items)}

    rows, cols = [], []
    row = 0
    for transaction, weight in db.weighted.items():
        kept = [column[item] for item in transaction if item in column]
        for _ in range(weight):
            rows.extend([row] * len(kept))
            cols.extend(kept)
            row += 1
    matrix = csr_matrix(([True] * len(rows), (rows, cols)), shape=(db.n_transactions, len(column)), dtype=bool)
    with warnings.catch_warnings():
        # pandas warns about the fill value of boolean sparse columns; the data is unaffected.
        warnings.simplefilter("ignore", FutureWarning)
        onehot = pd.DataFrame.sparse.from_spmatrix(matrix, columns=list(frequent_items))
    reference = mlxtend_fpgrowth(onehot, min_support=min_count / db.n_transactions, use_colnames=True)
    reference_counts = {
        tuple(sorted(itemset)): round(support * db.n_transactions)
        for itemset, support in zip(reference["itemsets"], reference["support"])
    }
    ours = _itemset_counts(result)
    return pd.Series(
        {
            "min_support": f"{result.min_support:.1%}",
            "our itemsets": len(ours),
            "mlxtend itemsets": len(reference_counts),
            "same itemsets": ours.keys() == reference_counts.keys(),
            "same support counts": ours == reference_counts,
        },
        name=f"{result.algorithm} vs mlxtend",
    )


def plot_performance(results: dict, rules: dict):
    """Frequent itemsets, association rules and runtime of both algorithms per minimum support."""
    table = performance_table(results, rules).reset_index()
    supports = list(dict.fromkeys(table["min_support"]))
    algorithms = list(dict.fromkeys(table["algorithm"]))
    panels = [
        ("frequent_itemsets", "Frequent itemsets", True),
        ("association_rules", "Association rules", True),
        ("runtime_s", "Median runtime (seconds)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    width = 0.8 / len(algorithms)
    for ax, (column, title, log) in zip(axes, panels):
        for i, algorithm in enumerate(algorithms):
            values = table[table["algorithm"] == algorithm].set_index("min_support").loc[supports, column]
            positions = [p + (i - (len(algorithms) - 1) / 2) * width for p in range(len(supports))]
            ax.bar(positions, values, width, color=COLORS.get(algorithm), label=algorithm)
            for x, value in zip(positions, values):
                label = f"{value:,.2f}" if column == "runtime_s" else f"{value:,}"
                ax.text(x, value, label, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(supports)), supports)
        if log:
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(0, table[column].max() * 3)
            title += " (log scale)"
        else:
            ax.set_ylim(0, table[column].max() * 1.12)
        ax.set(title=title, xlabel="Minimum support")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig
