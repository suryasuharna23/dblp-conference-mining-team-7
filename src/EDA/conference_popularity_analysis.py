"""Task 2 - Conference Popularity Analysis.

Popularity is measured by publication activity: the number of publications of a venue and
the number of unique authors who published there. Inputs are the `papers` and `authorship`
tables produced by Task 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

BAR_COLOR = "#3b6ea5"
ACCENT_COLOR = "#d98032"


# --------------------------------------------------------------------------------------
# Task 2 - conference popularity
# --------------------------------------------------------------------------------------
def venue_popularity(papers: pd.DataFrame, authorship: pd.DataFrame) -> pd.DataFrame:
    """Publications and unique authors of every venue, with the rank of each measure."""
    table = pd.DataFrame(
        {
            "publications": papers["venue"].value_counts(),
            "unique_authors": authorship.groupby("venue")["author"].nunique(),
        }
    )
    table["rank_publications"] = table["publications"].rank(ascending=False, method="min").astype(int)
    table["rank_authors"] = table["unique_authors"].rank(ascending=False, method="min").astype(int)
    table["authors_per_publication"] = (table["unique_authors"] / table["publications"]).round(2)
    return table.sort_values("publications", ascending=False).rename_axis("venue")


def top_by_publications(popularity: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return popularity.nsmallest(n, "rank_publications")[["publications", "rank_publications"]]


def top_by_unique_authors(popularity: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return popularity.nsmallest(n, "rank_authors")[["unique_authors", "rank_authors"]]


def compare_rankings(popularity: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Venues that are in the top n of either ranking, with both ranks side by side."""
    in_top = (popularity["rank_publications"] <= n) | (popularity["rank_authors"] <= n)
    table = popularity[in_top].copy()
    table["rank_shift"] = table["rank_publications"] - table["rank_authors"]
    return table.sort_values("rank_publications")


def ranking_agreement(popularity: pd.DataFrame, n: int = 10) -> pd.Series:
    """How similar the two rankings are: overlap of the top n and Spearman rank correlation."""
    top_pubs = set(popularity.nsmallest(n, "rank_publications").index)
    top_authors = set(popularity.nsmallest(n, "rank_authors").index)
    return pd.Series(
        {
            f"venues in both top {n}": len(top_pubs & top_authors),
            "spearman correlation (all venues)": round(
                popularity["publications"].corr(popularity["unique_authors"], method="spearman"), 4
            ),
        },
        name="value",
    )


def yearly_trend(papers: pd.DataFrame, venues: list[str]) -> pd.DataFrame:
    """Publications per year (rows) for the given venues (columns)."""
    subset = papers[papers["venue"].isin(venues)]
    table = subset.groupby(["year", "venue"]).size().unstack(fill_value=0)
    return table.reindex(columns=venues, fill_value=0)


def trend_summary(papers: pd.DataFrame, venues: list[str]) -> pd.DataFrame:
    """First year, peak year, last year and overall growth of every given venue.

    `years_without_data` counts the years in which DBLP lists no paper for the venue. A zero
    in the last year usually means the proceedings are not indexed yet, not that the
    conference stopped, so the growth of such a venue is left empty.
    """
    trend = yearly_trend(papers, venues)
    first, last = trend.iloc[0], trend.iloc[-1]
    return pd.DataFrame(
        {
            f"publications_{trend.index[0]}": first,
            f"publications_{trend.index[-1]}": last,
            "peak_year": trend.idxmax(),
            "peak_publications": trend.max(),
            "years_without_data": (trend == 0).sum(),
            "growth_%": ((last - first) / first.where(first > 0) * 100).where(last > 0).round(1),
        }
    )


def _barh(ax, series: pd.Series, title: str, xlabel: str, color: str) -> None:
    series = series.iloc[::-1]
    ax.barh(series.index, series.values, color=color)
    for y, value in enumerate(series.values):
        ax.text(value, y, f" {value:,}", va="center", fontsize=8)
    ax.set(title=title, xlabel=xlabel)
    ax.set_xlim(0, series.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)


def plot_top_venues(popularity: pd.DataFrame, n: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _barh(axes[0], popularity.nlargest(n, "publications")["publications"], f"Top {n} venues by publications", "Publications", BAR_COLOR)
    _barh(axes[1], popularity.nlargest(n, "unique_authors")["unique_authors"], f"Top {n} venues by unique authors", "Unique authors", ACCENT_COLOR)
    fig.tight_layout()
    return fig


def plot_yearly_trend(papers: pd.DataFrame, venues: list[str]):
    """Line per venue. A year without any indexed paper is drawn as a gap, not as a drop to zero."""
    trend = yearly_trend(papers, venues)
    trend = trend.where(trend > 0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = plt.get_cmap("tab10").colors
    for i, venue in enumerate(venues):
        ax.plot(trend.index, trend[venue], marker="o", markersize=4, linewidth=1.8, color=colors[i % 10], label=venue)
    ax.set(title="Publications per year of the top venues", xlabel="Year", ylabel="Publications")
    ax.set_xticks(trend.index)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ranking_comparison(popularity: pd.DataFrame, n: int = 10):
    """Slope chart: each line connects a venue's publication rank with its unique author rank."""
    table = compare_rankings(popularity, n)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for venue, rank_pubs, rank_authors in zip(table.index, table["rank_publications"], table["rank_authors"]):
        color = ACCENT_COLOR if rank_pubs != rank_authors else BAR_COLOR
        ax.plot([0, 1], [rank_pubs, rank_authors], marker="o", color=color)
        ax.text(-0.04, rank_pubs, f"{venue}  {rank_pubs}", ha="right", va="center", fontsize=9)
        ax.text(1.04, rank_authors, f"{rank_authors}  {venue}", ha="left", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xticks([0, 1], ["Rank by publications", "Rank by unique authors"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_title(f"Ranking comparison of venues in either top {n}")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    return fig
