"""
Plot solution from a results .txt file.

Usage:
    python plot_solution.py                 # plots results.txt
    python plot_solution.py results_gdp.txt # plots a specific file
    python plot_solution.py --no-show       # save to PNG without showing

Produces two figures:
    1. Gantt chart of the task schedule (one bar per unit)
    2. Bar chart of final production vs. demand per product
"""

import re
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

TASK_COLORS: dict[str, str] = {
    "Fa": "#E07B54",  # alcoholic fermentation
    "Fl": "#5B8DB8",  # lactic fermentation
    "Fr": "#6BBF7A",  # filtration
    "Alm": "#9B7EC8",  # storage
}

# Unit order for the y-axis
UNIT_ORDER: list[str] = [
    "inox5000",
    "inox10000",
    "inox17500",
    "inox22000",
    "subte6300",
    "subte7500",
    "subte9000",
    "subte9500",
    "subte12700",
    "iso5000",
    "iso10000",
]

UNIT_GROUP_LABELS: dict[str, str] = {
    "inox5000": "Inox tanks",
    "inox10000": "Inox tanks",
    "inox17500": "Inox tanks",
    "inox22000": "Inox tanks",
    "subte6300": "Subterranean tanks",
    "subte7500": "Subterranean tanks",
    "subte9000": "Subterranean tanks",
    "subte9500": "Subterranean tanks",
    "subte12700": "Subterranean tanks",
    "iso5000": "Isothermal tanks",
    "iso10000": "Isothermal tanks",
}


# Parser
def parse_results(filepath: str) -> dict:
    """
    Parse a results .txt file produced by utils.export_results
    and return a dict with keys:
        model_name, generated, objective, makespan, unused_units,
        tasks, production
    """
    text = Path(filepath).read_text(encoding="utf-8")

    meta = parse_metadata(text)

    # Prefer PLOTTING DATA if available
    plotting_section = re.search(
        r"PLOTTING DATA.*?\n[-]+\n(.*?)(?:\n={5,}|\Z)", text, re.DOTALL
    )
    if plotting_section:
        tasks = parse_txt(plotting_section.group(1))
    else:
        tasks = parse_task_schedule(text)

    meta["tasks"] = tasks
    meta["production"] = parse_production(text)
    return meta


def parse_metadata(text: str) -> dict:
    meta = {}

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("=") and i + 1 < len(lines):
            candidate = lines[i + 1].strip()
            if candidate and not candidate.startswith("="):
                meta["model_name"] = candidate
                break

    ts = re.search(r"Generated:\s*(.+)", text)
    meta["generated"] = ts.group(1).strip() if ts else ""

    obj = re.search(r"(?:Objective Value|Total Profit)\s*[:$\s]*([\d.,]+)", text)
    meta["objective"] = float(obj.group(1).replace(",", "")) if obj else None

    ms = re.search(r"Makespan:\s*([\d.]+)\s*hours", text)
    meta["makespan"] = float(ms.group(1)) if ms else None

    uu = re.search(r"Unused storage units:\s*(\d+)", text)
    meta["unused_units"] = int(uu.group(1)) if uu else None

    return meta


def parse_txt(section_text: str) -> list[dict]:
    tasks = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 5)
        if len(parts) < 5:
            continue
        task, event, start, end, batch = parts[:5]
        units_str = parts[5] if len(parts) > 5 else ""
        units = []
        for u in units_str.split(";"):
            u = u.strip()
            if ":" in u:
                uname, ubatch = u.rsplit(":", 1)
                units.append({"unit": uname.strip(), "batch": float(ubatch)})
        tasks.append(
            {
                "task": task.strip(),
                "event": int(event),
                "start": float(start),
                "end": float(end),
                "batch": float(batch),
                "units": units,
            }
        )
    return tasks


def parse_task_schedule(text: str) -> list[dict]:
    """Fallback parser for the TASK SCHEDULE section."""
    pattern = re.compile(
        r"Task\s+(\w+),\s+Event\s+(\d+):\s*\n"
        r"\s+Start:\s*([\d.]+)\s*h,\s*End:\s*([\d.]+)\s*h\s*\n"
        r"\s+Duration:.*?\n"
        r"\s+Batch:\s*([\d.]+)\s*L\s*\n"
        r"\s+Units:\s*(.*?)(?=\n\nTask|\n\n=|\Z)",
        re.DOTALL,
    )
    tasks = []
    for m in pattern.finditer(text):
        task, event, start, end, batch, units_raw = m.groups()
        units = [
            {"unit": u.group(1), "batch": float(u.group(2))}
            for u in re.finditer(r"(\w+)\s*\(([\d.]+)\s*L\)", units_raw)
        ]
        tasks.append(
            {
                "task": task,
                "event": int(event),
                "start": float(start),
                "end": float(end),
                "batch": float(batch),
                "units": units,
            }
        )
    return tasks


def parse_production(text: str) -> list[dict]:
    production = []
    for m in re.finditer(
        r"(p\d+|dsch):\s*([\d.]+)\s*L\s*\(Demand:\s*([\d.]+)\s*L\)", text
    ):
        production.append(
            {
                "product": m.group(1),
                "produced": float(m.group(2)),
                "demand": float(m.group(3)),
                # "outsourced": 0.0,
            }
        )
    if not production:
        for m in re.finditer(
            r"(p\d+|dsch):\s*Produced=([\d.]+)L,\s*Outsourced=([\d.]+)L,\s*Demand=([\d.]+)L",
            text,
        ):
            production.append(
                {
                    "product": m.group(1),
                    "produced": float(m.group(2)),
                    # "outsourced": float(m.group(3)),
                    "demand": float(m.group(4)),
                }
            )
    return production


def plot_solution(data: dict, ax: plt.Axes) -> None:
    tasks = data["tasks"]
    if not tasks:
        ax.text(
            0.5,
            0.5,
            "No task data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return

    used_units = []
    for t in tasks:
        for u in t["units"]:
            if u["unit"] not in used_units:
                used_units.append(u["unit"])

    # Order them according to UNIT_ORDER
    ordered_units = [u for u in UNIT_ORDER if u in used_units]
    ordered_units += [u for u in used_units if u not in ordered_units]

    unit_index = {u: i for i, u in enumerate(ordered_units)}
    bar_height = 0.6

    # Determine x-axis limit before drawing to skip out-of-range labels
    makespan = data.get("makespan") or max(t["end"] for t in tasks)
    xmax = makespan * 1.01

    for t in tasks:
        prefix = re.match(r"[A-Za-z]+", t["task"]).group()
        color = TASK_COLORS.get(prefix, "#AAAAAA")
        duration = t["end"] - t["start"]

        # Clip bar to xmax so it doesn't expand the axes
        visible_duration = min(t["end"], xmax) - t["start"]
        if visible_duration <= 0:
            continue

        for u in t["units"]:
            yi = unit_index.get(u["unit"])
            if yi is None:
                continue
            ax.barh(
                yi,
                visible_duration,
                left=t["start"],
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                align="center",
            )
            # For the lable, show task name + batch
            # but only if the bar centre is within xmax
            label_x = t["start"] + duration / 2
            if duration > 30 and label_x <= xmax:
                ax.text(
                    label_x,
                    yi,
                    f"{t['task']}\n{u['batch']:.0f} L",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )

    ax.set_yticks(range(len(ordered_units)))
    ax.set_yticklabels(ordered_units, fontsize=9)
    ax.set_ylim(-0.5, len(ordered_units) - 0.5)
    ax.invert_yaxis()
    ax.set_ylabel("Unit", fontsize=10)
    ax.set_axisbelow(True)

    # Secondary x-axis in days
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (days)", fontsize=10)
    ax2.xaxis.set_major_locator(MultipleLocator(7))

    # X-axis limits
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Time (hours)", fontsize=10)
    ax.xaxis.set_minor_locator(MultipleLocator(24))
    ax.xaxis.set_major_locator(MultipleLocator(168))
    ax2.set_xlim(0, xmax / 24)

    ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.grid(axis="x", which="major", linestyle="--", linewidth=0.6, alpha=0.5)

    legend_handles = [
        mpatches.Patch(facecolor=c, label=label)
        for label, c in [
            ("Alcoholic fermentation (Fa)", TASK_COLORS["Fa"]),
            ("Lactic fermentation (Fl)", TASK_COLORS["Fl"]),
            ("Filtration (Fr)", TASK_COLORS["Fr"]),
            ("Storage (Alm)", TASK_COLORS["Alm"]),
        ]
        if any(t["task"].startswith(label.split("(")[1].rstrip(")")) for t in tasks)
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)

    title_parts = []
    if data.get("model_name"):
        title_parts.append(data["model_name"])
    if data.get("makespan"):
        title_parts.append(
            f"Makespan: {data['makespan']:.0f} h ({data['makespan'] / 24:.1f} days)"
        )
    if data.get("objective") is not None:
        title_parts.append(f"Obj: {data['objective']:.2f}")
    if data.get("unused_units") is not None:
        title_parts.append(f"Unused tanks: {data['unused_units']}")
    ax.set_title("  |  ".join(title_parts), fontsize=10, pad=18)


# Production bar chart
def plot_production(data: dict, ax: plt.Axes) -> None:
    production = data.get("production", [])
    if not production:
        ax.text(
            0.5,
            0.5,
            "No production data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_title("Final Production", fontsize=11)
        return

    products = [p["product"] for p in production]
    produced = [p["produced"] for p in production]
    demand = [p["demand"] for p in production]
    # outsourced = [p.get("outsourced", 0.0) for p in production]

    x = range(len(products))
    width = 0.18
    gap = 0.02
    offset = width / 2 + gap / 2

    bars_prod = ax.bar(
        [xi - offset for xi in x],
        produced,
        width,
        label="Produced",
        color="#5B8DB8",
        edgecolor="white",
        linewidth=0.6,
    )
    # bars_out = ax.bar(
    #     [xi for xi in x],
    #     outsourced,
    #     width,
    #     label="Outsourced",
    #     color="#E07B54",
    #     edgecolor="white",
    #     linewidth=0.6,
    # )
    ax.bar(
        [xi + offset for xi in x],
        demand,
        width,
        label="Demand",
        color="#AAAAAA",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(products, fontsize=9)
    ax.set_ylabel("Volume (L)", fontsize=10)
    ax.set_title("Final Production vs. Demand", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    y_max = max(max(produced), max(demand))
    ax.set_ylim(0, y_max * 1.18)
    for bar in bars_prod:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + y_max * 0.01,
                f"{h:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot wine scheduling results from a txt file."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="results.txt",
        help="Path to the results txt file (default: results.txt)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure to PNG instead of displaying it",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG filename (default: <input_file>_gantt.png)",
    )
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)

    print(f"Parsing {filepath}...")
    data = parse_results(str(filepath))

    print(f"  Model    : {data.get('model_name', 'unknown')}")
    print(f"  Generated: {data.get('generated', 'N/A')}")
    print(f"  Tasks    : {len(data['tasks'])}")
    print(f"  Makespan : {data.get('makespan')} h")

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.025, 1, 1))
    has_production = bool(data.get("production"))
    if has_production:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax_gantt = fig.add_subplot(gs[0])
        ax_prod = fig.add_subplot(gs[1])
    else:
        ax_gantt = fig.add_subplot(111)
        ax_prod = None

    plot_solution(data, ax_gantt)
    if ax_prod is not None:
        plot_production(data, ax_prod)

    if data.get("generated"):
        fig.text(
            0.99,
            0.012,
            f"Generated: {data['generated']}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    if args.no_show or args.output:
        out_path = args.output or filepath.stem + "_gantt.png"
        fig.savefig(out_path, dpi=800, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
