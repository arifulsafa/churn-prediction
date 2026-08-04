"""Renders docs/architecture.png from the same description as docs/architecture.md.

Kept as code so the diagram cannot drift out of sync with the written design.
Run: python docs/make_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

COLOURS = {
    "source": "#dbe7f3",
    "process": "#cfe3d4",
    "train": "#f6e2c3",
    "registry": "#e4d7ef",
    "serve": "#f8d4d4",
    "consume": "#dcdcdc",
    "monitor": "#fdf1c0",
}

# (x, y, w, h, label, kind)
BOXES = [
    (0.4, 8.1, 2.5, 0.9, "Orders DB\n(Postgres)", "source"),
    (0.4, 7.0, 2.5, 0.9, "CRM / support\n(Zendesk)", "source"),
    (0.4, 5.9, 2.5, 0.9, "Web + app events\n(clickstream)", "source"),
    (0.4, 4.8, 2.5, 0.9, "Marketing\n(email, ads)", "source"),

    (3.7, 6.5, 2.6, 1.6, "Batch ELT (Airflow, daily)\nraw -> staged -> curated\ndbt tests on freshness\nand row counts", "process"),
    (3.7, 4.4, 2.6, 1.6, "Feature pipeline\npoint-in-time joins\nas of snapshot date\n-> Feature Store", "process"),

    (7.1, 6.8, 2.6, 1.3, "Training job (weekly)\nCV + tuning\ncost-based threshold", "train"),
    (7.1, 5.1, 2.6, 1.3, "Evaluation gate\nPR-AUC >= champion\nfairness + drift checks", "train"),

    (10.5, 6.0, 2.5, 1.5, "Model Registry\n(MLflow)\npipeline.joblib\n+ metadata + metrics", "registry"),

    (10.5, 3.6, 2.5, 1.4, "Prediction API\n(FastAPI, k8s)\n/predict /predict/batch", "serve"),
    (7.1, 3.6, 2.6, 1.4, "Nightly batch scoring\nfull customer base\n-> churn_scores table", "serve"),

    (13.8, 5.2, 2.4, 1.2, "CRM / retention\ncampaign tool", "consume"),
    (13.8, 3.6, 2.4, 1.2, "BI dashboard\n(cohorts, uplift)", "consume"),
    (13.8, 2.0, 2.4, 1.2, "Support desk\n(risk badge)", "consume"),

    (3.7, 1.4, 6.0, 1.4, "Monitoring: latency + error rate | input drift (PSI) | prediction drift |\n"
                          "outcome logging -> labels land 90 days later -> weekly performance report", "monitor"),
]

ARROWS = [
    ((2.9, 8.55), (3.7, 7.7)),
    ((2.9, 7.45), (3.7, 7.4)),
    ((2.9, 6.35), (3.7, 7.1)),
    ((2.9, 5.25), (3.7, 6.8)),
    ((5.0, 6.5), (5.0, 6.0)),          # ELT -> features
    ((6.3, 5.6), (7.1, 7.0)),          # features -> training
    ((8.4, 6.8), (8.4, 6.4)),          # training -> gate
    ((9.7, 5.7), (10.5, 6.4)),         # gate -> registry
    ((10.5, 6.2), (9.7, 4.6)),         # registry -> batch scoring
    ((11.75, 6.0), (11.75, 5.0)),      # registry -> API
    ((6.3, 4.9), (7.1, 4.4)),          # features -> batch scoring
    ((13.0, 4.4), (13.8, 5.6)),
    ((13.0, 4.3), (13.8, 4.2)),
    ((13.0, 4.1), (13.8, 2.8)),
    ((9.7, 3.9), (10.5, 3.9)),
]


def draw() -> Path:
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(1.0, 10.1)
    ax.axis("off")

    for x, y, w, h, label, kind in BOXES:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                facecolor=COLOURS[kind], edgecolor="#4a4a4a", linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)

    for start, end in ARROWS:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            color="#5a5a5a", linewidth=1.1, connectionstyle="arc3,rad=0.03")
        )

    # Online feature lookup, bowed under the batch-scoring box to avoid crossing it.
    ax.add_patch(
        FancyArrowPatch((6.3, 4.5), (10.5, 3.9), arrowstyle="-|>", mutation_scale=13,
                        color="#5a5a5a", linewidth=1.1, linestyle=(0, (4, 3)),
                        connectionstyle="arc3,rad=0.42")
    )
    ax.text(8.6, 3.08, "online feature lookup", fontsize=8.5, color="#5a5a5a", ha="center")

    # Feedback loop: outcomes observed 90 days later re-enter the warehouse.
    ax.add_patch(
        FancyArrowPatch((3.7, 2.1), (1.65, 4.8), arrowstyle="-|>", mutation_scale=13,
                        color="#b0553f", linewidth=1.4, linestyle="--",
                        connectionstyle="arc3,rad=0.35")
    )
    ax.text(1.5, 3.3, "observed outcomes\n(labels, T+90d)", fontsize=8.5, color="#b0553f", ha="center")

    ax.text(0.4, 9.75, "Churn prediction - production architecture", fontsize=16, weight="bold")
    ax.text(0.4, 9.42, "Daily batch features -> weekly retraining -> registry -> online + batch serving",
            fontsize=10.5, color="#444444")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                          markerfacecolor=c, markeredgecolor="#4a4a4a", label=k.title())
               for k, c in COLOURS.items()]
    ax.legend(handles=handles, loc="lower right", ncol=7, frameon=False, fontsize=9,
              bbox_to_anchor=(1.0, -0.02))

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(f"Wrote {draw()}")
