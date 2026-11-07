"""Redraw the figures without the legend row, with large fonts, in the paper
panel size (3.4 x 2.07 in, 600-dpi PNG + PDF).  The x-axis title is
centered on its axis when it fits; a title too long to center is slid
just enough to stay inside the figure (and clear of the other title).

Usage (in the folder with run_ten_figures.py):
    py replot_no_legend.py <run_dir>                    # all figures
    py replot_no_legend.py <run_dir> --figures 1,2,5,6,12,15
Output: <run_dir>/figures_nolegend/ plus one shared legend.png.
The original figures/, summary.csv and run_ten_figures.py are not touched.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import run_ten_figures as R

# Font sizes (pt).  Change these to make the text bigger or smaller.
R.PAPER_RC.update({"axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16,
                   "legend.fontsize": 10, "axes.linewidth": 1.2})
# Shorter x-axis titles (edit or remove entries as needed).
XLABEL = {
    "Avg. number of required features per task": "Avg. required features per task",
    "Avg. deadline (s)": "Avg. deadline",
}
PAD = 0.005  # gap to the figure edge, as a fraction of the figure
RENAME = {"figures_paper": "figures_nolegend", "summary.csv": "summary_nolegend.csv",
          "removed_outliers.csv": "removed_outliers_nolegend.csv"}

_savefig = plt.Figure.savefig


def _fit(fig, label, limit, axis):
    r = fig.canvas.get_renderer()
    while True:
        box = label.get_window_extent(r)
        size = box.width if axis == "x" else box.height
        if size <= limit or label.get_fontsize() <= 8:
            return
        label.set_fontsize(label.get_fontsize() - 0.5)


def _place(fig):
    """Same spot in every figure: the y title hugs the figure's left edge and
    the x title sits on the figure's bottom edge, so titles line up across
    panels.  Each title is centered on its axis when it fits; one too long
    to center is slid just enough to stay inside the figure."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for ax in fig.axes:
        pos = ax.get_position()
        xl, yl = ax.xaxis.label, ax.yaxis.label
        # y title: left edge at the figure edge, centered vertically on the axis
        yl.set_verticalalignment("top")      # rotated: 'top' faces left
        yl.set_horizontalalignment("center")
        h = yl.get_window_extent(r).height / H
        cy = pos.y0 + pos.height / 2
        cy = min(max(cy, PAD + h / 2), 1 - PAD - h / 2)
        ax.yaxis.set_label_coords((PAD - pos.x0) / pos.width, (cy - pos.y0) / pos.height)
        fig.canvas.draw()
        # x title: bottom at the figure edge, centered on the axis
        xl.set_verticalalignment("bottom")
        xl.set_horizontalalignment("center")
        w = xl.get_window_extent(r).width / W
        left = yl.get_window_extent(r).x1 / W + PAD
        cx = pos.x0 + pos.width / 2
        cx = min(max(cx, left + w / 2), 1 - PAD - w / 2)
        ax.xaxis.set_label_coords((cx - pos.x0) / pos.width, (PAD - pos.y0) / pos.height)


# Pass 1 measures every panel's tight layout; pass 2 gives all panels the
# same plot-box position (the tightest common one), so every figure has an
# identical axes size and the panels line up when placed side by side.
POSITIONS = []
FIXED = None


def _corner_savefig(fig, *a, **k):
    if a and str(a[0]).endswith(".pdf"):
        return None  # PNG only
    size = R.PAPER_RC["axes.labelsize"]
    for ax in fig.axes:
        xl, yl = ax.xaxis.label, ax.yaxis.label
        text = xl.get_text().replace("Average", "Avg.")
        xl.set_text(XLABEL.get(text, text))
        xl.set_fontsize(size)
        yl.set_fontsize(size)
        fig.canvas.draw()
        _fit(fig, yl, fig.bbox.height * (1 - 2 * PAD), "y")
        fig.canvas.draw()
        ywidth = yl.get_window_extent(fig.canvas.get_renderer()).width
        _fit(fig, xl, fig.bbox.width * (1 - 2 * PAD) - ywidth * 1.15, "x")
    if not fig.axes:  # the shared legend strip
        return _savefig(fig, *a, **k) if FIXED is not None else None
    if FIXED is None:
        fig.tight_layout(pad=0.2)
        for ax in fig.axes:
            p = ax.get_position()
            POSITIONS.append((p.x0, p.y0, p.x1, p.y1))
        return None
    for ax in fig.axes:
        ax.set_position(FIXED)
    _place(fig)
    fig.canvas.draw()
    return _savefig(fig, *a, **k)


plt.Figure.savefig = _corner_savefig


class RunDir(type(Path())):
    def __truediv__(self, key):
        return super().__truediv__(RENAME.get(str(key), key))


ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--figures", default="all")
ap.add_argument("--plot-runs", type=int, default=300)
args = ap.parse_args()

R.PLOT_OPTIONS.update(legend=False, paper=True, runs=args.plot_runs or None)
figs = list(R.FIGURES) if args.figures == "all" else [
    R.FIGURE_BY_NUMBER[int(i)] for i in args.figures.split(",")]
out = Path(args.run_dir) / "figures_nolegend"
(out / "legend.png").unlink(missing_ok=True)
R.plot(RunDir(args.run_dir), figs)          # pass 1: measure
x0 = max(p[0] for p in POSITIONS); y0 = max(p[1] for p in POSITIONS)
x1 = min(p[2] for p in POSITIONS); y1 = min(p[3] for p in POSITIONS)
FIXED = [x0, y0, x1 - x0, y1 - y0]
R.plot(RunDir(args.run_dir), figs)          # pass 2: draw with one common box
print("Saved to", out)
