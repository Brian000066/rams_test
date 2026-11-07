"""Bar chart of the default-setting ablation (output of run_ablation.py --default).

Each bar is one variant; the bar is stacked by cost item (activation,
inference, bandwidth, backhaul) and the total cost is written on top.
Same fonts as the paper figures (Times, bold, 600-dpi PNG).

Usage (in the folder with run_ablation.py):
    py plot_ablation_default.py                    # newest results/ablation/default_*
    py plot_ablation_default.py results\\ablation\\default_<time>
    py plot_ablation_default.py results\\ablation\\default_<time> --average own
Options:
    --average all   average over attempts where every variant is feasible (default)
    --average own   each variant averaged over its own feasible attempts
Output: <run dir>/ablation_default.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ("RESG-GoT", "SCOPE->Top-K", "w/o RSMA grouping", "w/o FAR", "CFRG->CSSGD",
         "CFRG->Location-aware")
TICK = {  # x-axis labels (two lines so they fit)
    "RESG-GoT": "RESG-GoT",
    "SCOPE->Top-K": "SCOPE\n" + r"$\rightarrow$Top-$K$",
    "w/o RSMA grouping": "w/o RSMA\ngrouping",
    "w/o FAR": "w/o\nFAR",
    "CFRG->CSSGD": "CFRG\n" + r"$\rightarrow$CSSGD",
    "CFRG->Location-aware": "CFRG\n" + r"$\rightarrow$Loc.",
}
ITEMS = (  # (column, legend label, color, hatch)
    ("activation_cost", "Activation", "#4C72B0", ""),
    ("inference_cost", "Inference", "#DD8452", "//"),
    ("bandwidth_cost", "Bandwidth", "#55A868", "xx"),
    ("backhaul_cost", "Backhaul", "#C44E52", ".."),
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 1.2,
})


def load(run_dir: Path, average: str) -> list[dict]:
    path = run_dir / "default_summary.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run  py run_ablation.py --default  first")
    want = "all feasible" if average == "all" else "own feasible"
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["average_over"] == want and r["total_cost"] != ""]
    for r in rows:  # older summaries call this column forwarding_cost
        r.setdefault("backhaul_cost", r.get("forwarding_cost", "0"))
    rows.sort(key=lambda r: ORDER.index(r["variant"]) if r["variant"] in ORDER else 99)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="?", default=None,
                        help="run directory (default: the newest results/ablation/default_*)")
    parser.add_argument("--average", choices=("all", "own"), default="all")
    args = parser.parse_args()
    if args.run_dir is None:
        root = Path(__file__).resolve().parent / "results" / "ablation"
        candidates = sorted(p for p in root.glob("default_*") if (p / "default_summary.csv").exists())
        if not candidates:
            raise SystemExit(f"No finished run under {root}; run  python3 run_ablation.py  first")
        args.run_dir = candidates[-1]
        print(f"Using {args.run_dir}")
    rows = load(args.run_dir, args.average)
    if not rows:
        raise SystemExit("no rows to plot")

    fig, ax = plt.subplots(figsize=(3.5 if len(rows) <= 4 else 5.0, 2.7))
    xs = range(len(rows))
    bottom = [0.0] * len(rows)
    for column, label, color, hatch in ITEMS:
        values = [float(r[column]) for r in rows]
        ax.bar(xs, values, 0.62, bottom=bottom, label=label, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6)
        bottom = [b + v for b, v in zip(bottom, values)]
    for x, r in zip(xs, rows):
        ax.text(x, float(r["total_cost"]) * 1.02, f"{float(r['total_cost']):.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([TICK.get(r["variant"], r["variant"]) for r in rows])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_ylabel("Total cost")
    ax.set_ylim(0, max(float(r["total_cost"]) for r in rows) * 1.12)
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.3, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    out = args.run_dir / "ablation_default.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved: {out}")
    for r in rows:
        print(f"  {r['variant']:20s} runs {r['runs']:>4}  total {float(r['total_cost']):8.1f}")


if __name__ == "__main__":
    main()
