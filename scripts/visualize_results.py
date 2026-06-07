"""Visualize RAG benchmark results as charts.

Usage:
    # From saved JSON report (load_test.py --json-output)
    python scripts/visualize_results.py --json benchmark_report.json

    # From inline data (manual entry)
    python scripts/visualize_results.py --inline

    # Specify output directory
    python scripts/visualize_results.py --inline --output-dir docs/img

Requirements:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)

# Configure matplotlib for Chinese text
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Dark theme colors matching the Gradio UI
_COLORS = {
    "vector": "#3b82f6",
    "bm25": "#22C55E",
    "hybrid": "#C9A84C",
    "hybrid_reranker": "#EAB308",
    "hybrid_graph": "#9333EA",
    "hybrid_correction": "#EF4444",
}

_COLOR_LIST = [
    "#3b82f6", "#22C55E", "#C9A84C", "#EAB308",
    "#9333EA", "#EF4444", "#06B6D4", "#F97316",
]


def _set_dark_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#0D1421")
    ax.tick_params(colors="#9CA3AF")
    ax.xaxis.label.set_color("#9CA3AF")
    ax.yaxis.label.set_color("#9CA3AF")
    ax.title.set_color("#E8E6E3")
    for spine in ax.spines.values():
        spine.set_color("#1E293B")


def plot_ragas_comparison(
    results: list[dict],
    output_dir: Path,
) -> None:
    """Plot grouped bar chart comparing RAG strategies across RAGAS metrics."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]

    strategies = [r["name"] for r in results]
    n_strategies = len(strategies)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0A0E27")
    _set_dark_style(ax)

    bar_width = 0.8 / n_strategies
    x = list(range(n_metrics))

    for i, strategy in enumerate(strategies):
        data = results[i]
        values = [data.get(m, 0) for m in metrics]
        color = _COLOR_LIST[i % len(_COLOR_LIST)]
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            bar_width,
            label=strategy,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.5,
            alpha=0.9,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, color="#9CA3AF",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RAG Strategy Comparison (RAGAS Metrics)", fontsize=14, pad=15)
    ax.legend(
        loc="upper right",
        fontsize=9,
        facecolor="#111832",
        edgecolor="#1E293B",
        labelcolor="#E8E6E3",
    )
    ax.grid(axis="y", alpha=0.15, color="#4B5563")

    plt.tight_layout()
    path = output_dir / "ragas_comparison.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency_comparison(
    results: list[dict],
    output_dir: Path,
) -> None:
    """Plot latency P50/P95 comparison."""
    strategies = [r["name"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0A0E27")
    _set_dark_style(ax)

    x = list(range(len(strategies)))
    width = 0.35

    p50s = [r.get("p50_ms", 0) / 1000 for r in results]
    p95s = [r.get("p95_ms", 0) / 1000 for r in results]

    bars1 = ax.bar(
        [xi - width / 2 for xi in x], p50s, width,
        label="P50", color="#3b82f6", edgecolor="#1E293B", alpha=0.9,
    )
    bars2 = ax.bar(
        [xi + width / 2 for xi in x], p95s, width,
        label="P95", color="#EAB308", edgecolor="#1E293B", alpha=0.9,
    )

    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}s", ha="center", fontsize=8, color="#9CA3AF",
            )
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}s", ha="center", fontsize=8, color="#9CA3AF",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title("Query Latency Comparison (P50 vs P95)", fontsize=14, pad=15)
    ax.legend(
        loc="upper right",
        fontsize=9,
        facecolor="#111832",
        edgecolor="#1E293B",
        labelcolor="#E8E6E3",
    )
    ax.grid(axis="y", alpha=0.15, color="#4B5563")

    plt.tight_layout()
    path = output_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_improvement_waterfall(
    results: list[dict],
    output_dir: Path,
) -> None:
    """Plot waterfall chart showing Faithfulness improvement across pipeline stages."""
    strategies = [r["name"] for r in results]
    faithfulness = [r.get("faithfulness", 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0A0E27")
    _set_dark_style(ax)

    colors = [_COLOR_LIST[i % len(_COLOR_LIST)] for i in range(len(strategies))]
    bars = ax.bar(strategies, faithfulness, color=colors, edgecolor="#1E293B", alpha=0.9)

    # Draw improvement annotations
    for i in range(1, len(faithfulness)):
        if faithfulness[i] > 0 and faithfulness[i - 1] > 0:
            diff = faithfulness[i] - faithfulness[i - 1]
            pct = diff / faithfulness[i - 1] * 100
            mid_x = (bars[i - 1].get_x() + bars[i].get_x() + bars[i].get_width()) / 2
            mid_y = (faithfulness[i] + faithfulness[i - 1]) / 2
            color = "#22C55E" if diff > 0 else "#EF4444"
            sign = "+" if diff > 0 else ""
            ax.annotate(
                f"{sign}{pct:.1f}%",
                xy=(mid_x, mid_y),
                fontsize=9, color=color,
                ha="center", va="center",
                fontweight="bold",
            )

    for bar, val in zip(bars, faithfulness):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, color="#E8E6E3",
            )

    ax.set_ylabel("Faithfulness Score", fontsize=11)
    ax.set_title("Faithfulness Improvement Across Pipeline Stages", fontsize=14, pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.15, color="#4B5563")

    plt.tight_layout()
    path = output_dir / "faithfulness_waterfall.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def generate_default_data() -> list[dict]:
    """Return Round 9 benchmark data as default."""
    return [
        {
            "name": "Vector",
            "faithfulness": 0.8010,
            "answer_relevancy": 0.3218,
            "context_precision": 0.7034,
            "context_recall": 0.6215,
            "p50_ms": 25858,
            "p95_ms": 56802,
        },
        {
            "name": "BM25",
            "faithfulness": 0.7952,
            "answer_relevancy": 0.3511,
            "context_precision": 0.7110,
            "context_recall": 0.5746,
        },
        {
            "name": "Hybrid",
            "faithfulness": 0.6609,
            "answer_relevancy": 0.3267,
            "context_precision": 0.7302,
            "context_recall": 0.6091,
            "p50_ms": 21698,
            "p95_ms": 54346,
        },
        {
            "name": "Hybrid+Reranker",
            "faithfulness": 0.7623,
            "answer_relevancy": 0.3298,
            "context_precision": 0.6995,
            "context_recall": 0.6341,
        },
        {
            "name": "Hybrid+Graph",
            "faithfulness": 0.6909,
            "answer_relevancy": 0.3251,
            "context_precision": 0.6988,
            "context_recall": 0.5985,
        },
        {
            "name": "Hybrid+SC",
            "faithfulness": 0.8162,
            "answer_relevancy": 0.3278,
            "context_precision": 0.7036,
            "context_recall": 0.6188,
            "p50_ms": 19221,
            "p95_ms": 60306,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Visualize RAG benchmark results")
    parser.add_argument(
        "--json", type=str, default="",
        help="JSON file with benchmark results",
    )
    parser.add_argument(
        "--inline", action="store_true",
        help="Use built-in Round 9 data for demo",
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/img",
        help="Output directory for charts (default: docs/img)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
        results = data if isinstance(data, list) else data.get("results", [])
    elif args.inline:
        results = generate_default_data()
    else:
        print("Specify --json <file> or --inline for demo data")
        sys.exit(1)

    if not results:
        print("No data to visualize")
        sys.exit(1)

    print(f"Visualizing {len(results)} strategies...")

    plot_ragas_comparison(results, output_dir)
    plot_latency_comparison(results, output_dir)
    plot_improvement_waterfall(results, output_dir)

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
