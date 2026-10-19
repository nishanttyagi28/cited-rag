"""Draw benchmark charts from results/*.json into docs/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, DOCS = ROOT / "results", ROOT / "docs"


def retrieval_chart(bench: dict) -> None:
    metrics = ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10"]
    configs = list(bench["retrieval"])
    width = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, c in enumerate(configs):
        vals = [bench["retrieval"][c][m] for m in metrics]
        xs = [j + i * width for j in range(len(metrics))]
        bars = ax.bar(xs, vals, width, label=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks([j + width * (len(configs) - 1) / 2 for j in range(len(metrics))], metrics)
    ax.set_ylim(0, 1.08)
    n = next(iter(bench["retrieval"].values()))["n_answerable"]
    ax.set_title(f"Retrieval quality on {n} answerable golden questions (chart)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "chart_retrieval.png", dpi=130)


def kind_chart(bench: dict) -> None:
    configs = list(bench["retrieval"])
    kinds = sorted(next(iter(bench["retrieval"].values()))["by_kind"])
    width = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, c in enumerate(configs):
        bk = bench["retrieval"][c]["by_kind"]
        ax.bar(
            [j + i * width for j in range(len(kinds))],
            [bk[k]["mrr@10"] for k in kinds],
            width,
            label=c,
        )
    labels = [f"{k} (n={bench['retrieval'][configs[0]]['by_kind'][k]['n']})" for k in kinds]
    ax.set_xticks([j + width * (len(configs) - 1) / 2 for j in range(len(kinds))], labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("MRR@10 by question kind (chart)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "chart_by_kind.png", dpi=130)


def answer_chart(bench: dict) -> None:
    ans = bench.get("answers")
    if not ans:
        return
    metrics = ["key_fact_recall", "citation_precision", "sentence_support_rate", "refusal_accuracy"]
    configs = list(ans)
    width = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, c in enumerate(configs):
        bars = ax.bar(
            [j + i * width for j in range(len(metrics))],
            [ans[c][m] for m in metrics],
            width,
            label=c,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(
        [j + width * (len(configs) - 1) / 2 for j in range(len(metrics))], metrics, fontsize=8
    )
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Answer-level metrics, {next(iter(ans.values()))['model']} (chart)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "chart_answers.png", dpi=130)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    bench = json.loads((RES / "benchmark.json").read_text())
    retrieval_chart(bench)
    kind_chart(bench)
    answer_chart(bench)
    print("charts written to", DOCS)


if __name__ == "__main__":
    main()
