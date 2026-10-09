"""Render the before and after figure from the measured metrics."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_RULE = "#c3c2b7"

NEUTRAL = "#898781"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

SERIES = [
    ("ungated", "Clone every polymorphic function", ORANGE),
    ("ablation", "Ablation: one shared clone", AQUA),
    ("monoclone", "monoclone (cost model gated)", BLUE),
]


def _style(axis):
    axis.set_facecolor(SURFACE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(BASELINE_RULE)
    axis.spines["bottom"].set_color(BASELINE_RULE)
    axis.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axis.set_axisbelow(True)


def _panel_total(axis, metrics):
    totals = metrics["summary"]["total_ms"]
    speedup = metrics["summary"]["speedup"]
    labels = ["Baseline\nno cloning", "Clone every\npolymorphic function", "monoclone\ncost model gated"]
    values = [totals["baseline"], totals["ungated"], totals["monoclone"]]
    colors = [NEUTRAL, ORANGE, BLUE]
    positions = range(len(values))
    bars = axis.bar(positions, values, width=0.56, color=colors, linewidth=0)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    for index, (bar, value) in enumerate(zip(bars, values)):
        tag = "%.1f ms" % value
        if index:
            key = ["baseline", "ungated", "monoclone"][index]
            tag += "\n%.2fx" % speedup[key]
        axis.annotate(
            tag,
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9.5,
            color=INK,
            linespacing=1.35,
        )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY, linespacing=1.3)
    axis.set_ylabel(
        "total wall clock, %d workloads (ms)" % len(metrics["workloads"]),
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    axis.set_ylim(0, max(values) * 1.26)
    axis.set_title(
        "Same work, three policies", fontsize=11.5, color=INK, loc="left", pad=30
    )
    _style(axis)


def _panel_per_workload(axis, metrics):
    records = metrics["workloads"]
    names = [r["name"] for r in records]
    count = len(records)
    width = 0.26
    noise = metrics["summary"]["noise_floor_pct"] / 100.0

    axis.axhspan(1 - noise, 1 + noise, color=GRID, linewidth=0, zorder=0)
    axis.axhline(1.0, color=BASELINE_RULE, linewidth=1.2, linestyle="--", zorder=1)

    for offset, (key, label, color) in enumerate(SERIES):
        positions = [i + (offset - 1) * width for i in range(count)]
        values = [r["speedup"][key] for r in records]
        axis.bar(
            positions,
            values,
            width=width - 0.02,
            color=color,
            label=label,
            linewidth=0,
            zorder=2,
        )
        for x, value in zip(positions, values):
            axis.annotate(
                "%.2f" % value,
                (x, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.4,
                color=INK_SECONDARY,
                rotation=90,
                zorder=3,
            )

    for index, record in enumerate(records):
        if not record["cloned"]:
            axis.annotate(
                "left alone",
                (index, 1.47),
                ha="center",
                fontsize=8.0,
                color=INK_MUTED,
                zorder=3,
            )

    axis.set_xticks(range(count))
    axis.set_xticklabels(names, fontsize=8.6, color=INK_SECONDARY, rotation=18, ha="right")
    axis.set_ylabel("speedup vs baseline", fontsize=9.5, color=INK_SECONDARY)
    axis.set_ylim(0, 1.58)
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_title(
        "Per workload, 1.00 is the untouched function",
        fontsize=11.5,
        color=INK,
        loc="left",
        pad=30,
    )
    legend = axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.055),
        ncol=3,
        frameon=False,
        fontsize=8.8,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
        borderpad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _style(axis)


def render(metrics, path):
    figure, axes = plt.subplots(
        1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1.0, 1.75]}
    )
    figure.patch.set_facecolor(SURFACE)
    _panel_total(axes[0], metrics)
    _panel_per_workload(axes[1], metrics)
    summary = metrics["summary"]
    figure.suptitle(
        "Removing call site specialization thrash in CPython %s" % metrics["python"],
        fontsize=13.5,
        color=INK,
        x=0.012,
        ha="left",
        y=0.975,
    )
    figure.text(
        0.012,
        0.036,
        "Minimum of %d interleaved repeats on CPython %s. The shaded band is the "
        "measured noise floor (%.1f percent): the largest deviation from 1.00 among "
        "the workloads left alone,"
        % (metrics["reps"], metrics["python"], summary["noise_floor_pct"]),
        fontsize=8.2,
        color=INK_MUTED,
    )
    figure.text(
        0.012,
        0.010,
        "where the measured callable is the original function object itself. The "
        "ablation pays the same dispatch and removes only monomorphism, so what it "
        "gives up is exactly what specialization was worth.",
        fontsize=8.2,
        color=INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.945))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
