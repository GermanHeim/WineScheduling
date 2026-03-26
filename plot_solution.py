"""
Plot solution from a results .txt file.

Usage:
    python plot_solution.py                 # plots results.txt
    python plot_solution.py results_gdp.txt # plots a specific file
    python plot_solution.py --no-show       # save to PNG without showing
    python plot_solution.py --stn --no-show # also plot the STN graph
    python plot_solution.py --publish       # save to EPS (publication mode)

Produces two figures:
    1. Gantt chart of the task schedule (one bar per unit)
    2. Bar chart of final production vs. demand per product
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

TASK_COLORS: dict[str, str] = {
    "Pr": "#C06C84",  # pressing
    "Fa": "#E07B54",  # alcoholic fermentation
    "Fl": "#5B8DB8",  # lactic fermentation
    "Cs": "#6BBF7A",  # cold stabilization
    "Alm": "#9B7EC8",  # storage
}

# Unit order for the y-axis
UNIT_ORDER: list[str] = [
    "press",
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
    "press": "Press",
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
    meta["wine_metadata"] = parse_wine_metadata(text)
    meta["production"] = parse_production(text)
    return meta


def parse_metadata(text: str) -> dict:
    meta: dict[str, object] = {}

    def parse_metric(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1).replace(",", "")) if m else None

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

    meta["revenue"] = parse_metric(r"Revenue\s*[:$\s]*([\d.,]+)")
    meta["outsourcing_cost"] = parse_metric(r"Outsourcing Cost\s*[:$\s]*([\d.,]+)")
    meta["raw_material_cost"] = parse_metric(r"Raw Material Cost\s*[:$\s]*([\d.,]+)")
    meta["lateness_cost"] = parse_metric(r"Lateness Cost\s*[:$\s]*([\d.,]+)")

    ms = re.search(r"Makespan:\s*([\d.]+)\s*hours", text)
    meta["makespan"] = float(ms.group(1)) if ms else None

    uu = re.search(r"Unused storage units:\s*(\d+)", text)
    meta["unused_units"] = int(uu.group(1)) if uu else None

    return meta


def parse_wine_metadata(text: str) -> dict[str, dict[str, str]]:
    wine_map: dict[str, dict[str, str]] = {}
    section = re.search(r"WINE METADATA.*?\n[-]+\n(.*?)(?:\n={5,}|\Z)", text, re.DOTALL)
    if not section:
        return wine_map

    for line in section.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        _, product, name, category = parts
        wine_map[product] = {"name": name, "category": category}

    return wine_map


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
    line_pattern = re.compile(
        r"^([A-Za-z0-9_]+):\s*([\d.]+)\s*L\s*\(Demand:\s*([\d.]+)\s*L\)"
        r"(?:\s*\[(.*?)\])?(?:,\s*Outsourced:\s*([\d.]+)\s*L)?\s*$"
    )
    for line in text.splitlines():
        m = line_pattern.match(line.strip())
        if not m:
            continue
        wine_name = ""
        category = ""
        if m.group(4):
            parts = [part.strip() for part in m.group(4).split("|", 1)]
            if len(parts) == 2:
                wine_name, category = parts
        outsourced = float(m.group(5)) if m.group(5) else 0.0
        production.append(
            {
                "product": m.group(1),
                "produced": float(m.group(2)),
                "outsourced": outsourced,
                "demand": float(m.group(3)),
                "name": wine_name,
                "category": category,
            }
        )

    outsourced_by_product: dict[str, float] = {}
    produced_out_demand_matches = list(
        re.finditer(
            r"([A-Za-z0-9_]+):\s*Produced=([\d.]+)L,\s*Outsourced=([\d.]+)L,\s*Demand=([\d.]+)L",
            text,
        )
    )

    for m in produced_out_demand_matches:
        product = m.group(1)
        outsourced_by_product[product] = float(m.group(3))

    if production and outsourced_by_product:
        for item in production:
            product_key = str(item.get("product", ""))
            item["outsourced"] = outsourced_by_product.get(product_key, 0.0)
    else:
        for m in produced_out_demand_matches:
            production.append(
                {
                    "product": m.group(1),
                    "produced": float(m.group(2)),
                    "outsourced": float(m.group(3)),
                    "demand": float(m.group(4)),
                }
            )

    return production


def infer_toml_path(results_path: Path, model_name: str) -> Path:
    """Infer which TOML file should be used for STN plotting."""
    model_l = (model_name or "").lower()
    file_l = results_path.name.lower()

    if "economic" in model_l or "economic" in file_l:
        preferred = Path("parameters.toml")
        if preferred.exists():
            return preferred

    if "ms" in model_l or "ms" in file_l:
        preferred = Path("parametersMS.toml")
        if preferred.exists():
            return preferred

    # Historical default for the non-economic model.
    default_ms = Path("parametersMS.toml")
    if default_ms.exists():
        return default_ms

    fallback = Path("parameters.toml")
    if fallback.exists():
        return fallback

    # Final fallback keeps previous behavior when files are missing.
    return default_ms


def is_economic_context(results_path: Path, model_name: str) -> bool:
    model_l = (model_name or "").lower()
    file_l = results_path.name.lower()
    return "economic" in model_l or "economic" in file_l


def read_deadline_hours(toml_path: Path) -> float | None:
    if not toml_path.exists():
        return None
    with open(toml_path, "rb") as f:
        params = tomllib.load(f)
    deadline = params.get("global", {}).get("deadline")
    if deadline is None:
        return None
    try:
        return float(deadline)
    except (TypeError, ValueError):
        return None


def task_stage(task_name: str) -> str:
    return "".join(ch for ch in task_name if ch.isalpha())


def task_line(task_name: str) -> str:
    return "".join(ch for ch in task_name if ch.isdigit())


def format_task_label(task_name: str, publish_mode: bool) -> str:
    """Return a display label for task codes, expanded in publish mode."""
    if not publish_mode:
        return task_name

    stage = task_stage(task_name)
    line = task_line(task_name)
    long_names = {
        "Fa": "Alcoholic Fermentation",
        "Fl": "Malolactic Fermentation",
        "Alm": "Storage",
        "Cs": "Cold Stabilization",
    }
    if stage in long_names and line:
        return f"{long_names[stage]} ({line})"
    return task_name


def format_unit_label(unit_name: str, publish_mode: bool) -> str:
    """Return a display label for units, expanded in publish mode."""
    if not publish_mode:
        return unit_name

    m = re.match(r"^(inox|subte|iso)(\d+)$", unit_name, re.IGNORECASE)
    if not m:
        return unit_name

    family = m.group(1).lower().capitalize()
    capacity = m.group(2)
    return f"{family} {capacity}"


def parse_stn_data(toml_file: str) -> dict:
    with open(toml_file, "rb") as f:
        params = tomllib.load(f)

    lines_cfg = params.get("lines", {})
    products = params.get("products", {})
    initial_inventory = params.get("initial_inventory", {})
    product_ub = params.get("product_ub", {})
    rho_cfg = params.get("rho", {})

    tasks = []
    line_product = {}
    for line, cfg in lines_cfg.items():
        line_product[line] = cfg["product"]
        for step in cfg["steps"]:
            tasks.append(f"{step}{line}")

    states: set[str] = {"dsch"}
    for line, cfg in lines_cfg.items():
        if "Pr" in cfg["steps"]:
            states.add(f"m{line}")
        states.add(f"v{line}")
        if "Fl" in cfg["steps"]:
            states.add(f"vl{line}")
        states.add(cfg["product"])

    # Include any explicitly declared inventory states. This is important when
    # raw materials are shared pools (e.g. s_red, s_white_rose) instead of s1..sN.
    states.update(initial_inventory.keys())

    arcs = []
    for task in tasks:
        rho = rho_cfg.get(task, {})
        for state, value in rho.get("rhoIScons", {}).items():
            if value < 0:
                arcs.append(
                    {
                        "src": state,
                        "dst": task,
                        "coef": abs(float(value)),
                        "kind": "cons",
                    }
                )
                states.add(state)
        for state, value in rho.get("rhoISprod", {}).items():
            if value > 0:
                arcs.append(
                    {
                        "src": task,
                        "dst": state,
                        "coef": float(value),
                        "kind": "prod",
                    }
                )
                states.add(state)

    return {
        "tasks": tasks,
        "states": sorted(states),
        "arcs": arcs,
        "lines_cfg": lines_cfg,
        "line_product": line_product,
        "products": products,
        "initial_inventory": initial_inventory,
        "product_ub": product_ub,
    }


def format_arc_label(value: float) -> str:
    if 0 <= value <= 1.5:
        return f"{value * 100:.0f}%"
    return f"{value:.2f}"


def build_stn_positions(stn_data: dict) -> dict[str, tuple[float, float]]:
    lines_cfg = stn_data["lines_cfg"]
    tasks = stn_data["tasks"]
    states = stn_data["states"]
    line_product = stn_data["line_product"]
    arcs = stn_data["arcs"]

    sorted_lines = sorted(lines_cfg.keys(), key=lambda x: int(x))
    y_by_line = {line: -idx for idx, line in enumerate(sorted_lines)}

    stage_x = {
        "Pr": 1.3,
        "Fa": 2.3,
        "Fl": 3.4,
        "Cs": 4.5,
        "Alm": 5.6,
    }

    # Place material states between the stages that consume/produce them to preserve
    # the visual causal direction (e.g.: s -> Pr -> m -> Fa).
    state_x = {
        "s": 0.2,
        "m": 1.8,
        "v": 2.95,
        "vl": 3.95,
        "p": 6.6,
    }

    pos: dict[str, tuple[float, float]] = {}

    for task in tasks:
        line = task_line(task)
        stage = task_stage(task)
        pos[task] = (stage_x.get(stage, 3.0), y_by_line[line])

    task_line_map = {task: task_line(task) for task in tasks}

    def state_connected_lines(state: str) -> list[str]:
        connected: list[str] = []
        for arc in arcs:
            if arc["src"] == state and arc["dst"] in task_line_map:
                connected.append(task_line_map[arc["dst"]])
            elif arc["dst"] == state and arc["src"] in task_line_map:
                connected.append(task_line_map[arc["src"]])
        return [line for line in connected if line in y_by_line]

    for state in states:
        if state == "dsch":
            pos[state] = (6.8, 1.2)
            continue

        y_values: list[float] = []

        # Prefer explicit product-to-line mapping for final product states.
        product_line = ""
        for candidate, product in line_product.items():
            if product == state:
                product_line = candidate
                break
        if product_line in y_by_line:
            y_values.append(y_by_line[product_line])

        # Add connected task lines so shared states (e.g. s_red) are centered
        # across all relevant lines instead of forced to a single line index.
        for line in state_connected_lines(state):
            y_values.append(y_by_line[line])

        if y_values:
            y = sum(y_values) / len(y_values)
        else:
            y = 0.0

        if state.startswith("s"):
            pos[state] = (state_x["s"], y)
        elif state.startswith("m"):
            pos[state] = (state_x["m"], y)
        elif state.startswith("vl"):
            pos[state] = (state_x["vl"], y)
        elif state.startswith("v"):
            pos[state] = (state_x["v"], y)
        else:
            pos[state] = (state_x["p"], y)

    return pos


def plot_stn_graph(toml_file: str, ax: plt.Axes, show_dsch: bool = True) -> None:
    stn_data = parse_stn_data(toml_file)
    pos = build_stn_positions(stn_data)
    products = stn_data["products"]
    initial_inventory = stn_data["initial_inventory"]
    product_ub = stn_data["product_ub"]

    state_nodes = [state for state in stn_data["states"] if state != "dsch"]
    task_nodes = stn_data["tasks"]

    category_color = {
        "Red": "#C94F4F",
        "White": "#D9D9D9",
        "Rose": "#F29BB2",
    }

    # Draw arcs first so nodes stay on top
    for arc in stn_data["arcs"]:
        # Avoid visual clutter: show each loss-to-dsch as a local upward arrow
        # from the producing task instead of routing every arc to one shared dsch node.
        if (
            show_dsch
            and arc["kind"] == "prod"
            and arc["dst"] == "dsch"
            and arc["src"] in task_nodes
        ):
            x_task, y_task = pos[arc["src"]]
            y_top = y_task + 0.72
            patch = mpatches.FancyArrowPatch(
                (x_task, y_task + 0.03),
                (x_task, y_top),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.5,
                color="#C0392B",
                linestyle="-",
                alpha=0.9,
                connectionstyle="arc3,rad=0.0",
                zorder=1,
            )
            ax.add_patch(patch)
            ax.text(
                x_task,
                y_top + 0.05,
                f"dsch {format_arc_label(arc['coef'])}",
                fontsize=7,
                color="#C0392B",
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8,
                },
                zorder=2,
            )
            continue

        if (not show_dsch) and arc["dst"] == "dsch":
            continue

        x1, y1 = pos[arc["src"]]
        x2, y2 = pos[arc["dst"]]
        color = "#2A9D8F" if arc["kind"] == "prod" else "#555555"
        linestyle = "-" if arc["kind"] == "prod" else "--"
        patch = mpatches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.5,
            color=color,
            linestyle=linestyle,
            alpha=0.85,
            connectionstyle="arc3,rad=0.08",
            zorder=1,
        )
        ax.add_patch(patch)

        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2 + (0.08 if arc["kind"] == "prod" else -0.08)
        ax.text(
            label_x,
            label_y,
            format_arc_label(arc["coef"]),
            fontsize=7,
            color=color,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
            zorder=2,
        )

    # Draw state nodes
    for state in state_nodes:
        x, y = pos[state]
        if state.startswith("s"):
            marker = "o"
            node_color = "#AED6F1"
        elif state.startswith("m"):
            marker = "o"
            node_color = "#FAD7A0"
        elif state.startswith(("v", "vl")):
            marker = "o"
            node_color = "#ABEBC6"
        else:
            marker = "s"
            cat = products.get(state, {}).get("category", "")
            node_color = category_color.get(cat, "#D5DBDB")

        has_capacity_data = (state in initial_inventory) or (state in product_ub)
        edge_color = "#1B4F72" if has_capacity_data else "#424949"
        edge_width = 2.0 if has_capacity_data else 1.0

        ax.scatter(
            [x],
            [y],
            s=230,
            marker=marker,
            c=node_color,
            edgecolors=edge_color,
            linewidths=edge_width,
            zorder=3,
        )

        label = state
        if state in products:
            name = products[state].get("name", state)
            label = f"{state}\n{name}"

        cap_parts = []
        if state in initial_inventory:
            cap_parts.append(f"Init={initial_inventory[state]:.0f}L")
        if state in product_ub:
            cap_parts.append(f"UB={product_ub[state]:.0f}L")
        if cap_parts:
            label = label + "\n" + " | ".join(cap_parts)

        ax.text(x, y - 0.36, label, fontsize=7, ha="center", va="top", zorder=4)

    # Draw task nodes
    for task in task_nodes:
        x, y = pos[task]
        stage = task_stage(task)
        ax.scatter(
            [x],
            [y],
            s=200,
            marker="^",
            c=TASK_COLORS.get(stage, "#BBBBBB"),
            edgecolors="#2D3436",
            linewidths=1.0,
            zorder=4,
        )
        ax.text(x, y + 0.24, task, fontsize=7, ha="center", va="bottom", zorder=5)

    ax.set_title(
        "State-Task Network (STN): States, Tasks, Material Flows, and Yield Fractions",
        fontsize=11,
    )
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(-len(stn_data["lines_cfg"]) - 0.8, 2.1)
    ax.axis("off")

    legend_items = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#424949", label="Raw state (S)"),
        mpatches.Patch(
            facecolor="#ABEBC6", edgecolor="#424949", label="Intermediate state"
        ),
        mpatches.Patch(facecolor="#D5DBDB", edgecolor="#424949", label="Final state"),
        mpatches.Patch(
            facecolor="#FFFFFF",
            edgecolor="#1B4F72",
            label="Bold border: has Init/UB data",
        ),
        mpatches.Patch(
            facecolor="#FFFFFF", edgecolor="#2A9D8F", label="Solid arc: production"
        ),
        mpatches.Patch(
            facecolor="#FFFFFF", edgecolor="#555555", label="Dashed arc: consumption"
        ),
    ]
    if show_dsch:
        legend_items.append(
            mpatches.Patch(
                facecolor="#FFFFFF",
                edgecolor="#C0392B",
                label="Upward red arrow: loss to dsch",
            )
        )
    ax.legend(handles=legend_items, loc="lower left", fontsize=7, framealpha=0.9)


def plot_solution(
    data: dict,
    ax: plt.Axes,
    deadline_hours: float | None = None,
    publish_mode: bool = False,
) -> None:
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
        match = re.match(r"[A-Za-z]+", t["task"])
        prefix = match.group() if match else ""
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
                display_task = format_task_label(t["task"], publish_mode)
                ax.text(
                    label_x,
                    yi,
                    f"{display_task}\n{u['batch']:.0f} L",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )

    ax.set_yticks(range(len(ordered_units)))
    display_units = [format_unit_label(u, publish_mode) for u in ordered_units]
    ax.set_yticklabels(display_units, fontsize=9)
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

    if deadline_hours is not None and deadline_hours >= 0:
        ax.axvline(
            deadline_hours,
            color="#B03A2E",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            zorder=0,
        )

    legend_defs = [
        ("Pr", "Pressing", TASK_COLORS["Pr"]),
        ("Fa", "Alcoholic fermentation", TASK_COLORS["Fa"]),
        ("Fl", "Lactic fermentation", TASK_COLORS["Fl"]),
        ("Cs", "Cold stabilization", TASK_COLORS["Cs"]),
        ("Alm", "Storage", TASK_COLORS["Alm"]),
    ]
    legend_handles: list[Artist] = [
        mpatches.Patch(
            facecolor=color,
            label=name if publish_mode else f"{name} ({code})",
        )
        for code, name, color in legend_defs
        if any(t["task"].startswith(code) for t in tasks)
    ]
    if deadline_hours is not None and deadline_hours >= 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#B03A2E",
                linestyle="--",
                linewidth=1.4,
                label=f"Deadline ({deadline_hours:.0f} h)",
            )
        )
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
    if data.get("revenue") is not None:
        title_parts.append(f"Revenue: {data['revenue']:.2f}")
    model_name_l = str(data.get("model_name", "")).lower()
    if "economic" in model_name_l:
        econ_parts = []
        if data.get("outsourcing_cost") is not None:
            econ_parts.append(f"Outsource Cost: {data['outsourcing_cost']:.2f}")
        if data.get("raw_material_cost") is not None:
            econ_parts.append(f"Raw Mat Cost: {data['raw_material_cost']:.2f}")
        if data.get("lateness_cost") is not None:
            econ_parts.append(f"Lateness Cost: {data['lateness_cost']:.2f}")
        if econ_parts:
            title_parts.append("; ".join(econ_parts))
    if data.get("unused_units") is not None:
        title_parts.append(f"Unused tanks: {data['unused_units']}")
    ax.set_title("  |  ".join(title_parts), fontsize=10, pad=18)


# Production bar chart
def plot_production(data: dict, ax: plt.Axes, show_dsch: bool = True) -> None:
    production = data.get("production", [])
    if not show_dsch:
        production = [p for p in production if p.get("product", "").lower() != "dsch"]

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

    wine_metadata = data.get("wine_metadata", {})
    products = []
    for p in production:
        product_key = p["product"]
        meta = wine_metadata.get(product_key, {})
        wine_name = p.get("name") or meta.get("name", "")
        category = p.get("category") or meta.get("category", "")
        if wine_name and category:
            products.append(f"{wine_name}\n({category})")
        elif wine_name:
            products.append(wine_name)
        else:
            products.append(product_key)
    produced = [p["produced"] for p in production]
    demand = [p["demand"] for p in production]
    outsourced = [p.get("outsourced", 0.0) for p in production]
    show_outsourced = any(v > 1e-9 for v in outsourced)

    x = range(len(products))
    if show_outsourced:
        width = 0.16
        gap = 0.02
        delta = width + gap
        ax.bar(
            [xi - delta for xi in x],
            outsourced,
            width,
            label="Outsourced",
            color="#E07B54",
            edgecolor="white",
            linewidth=0.6,
        )
        bars_prod = ax.bar(
            [xi for xi in x],
            produced,
            width,
            label="Produced",
            color="#5B8DB8",
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            [xi + delta for xi in x],
            demand,
            width,
            label="Demand",
            color="#AAAAAA",
            edgecolor="white",
            linewidth=0.6,
        )
    else:
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
    if show_outsourced:
        ax.set_title("Final Production vs. Outsourced vs. Demand", fontsize=10)
    else:
        ax.set_title("Final Production vs. Demand", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    y_max = max(max(produced), max(demand), max(outsourced) if outsourced else 0.0)
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


def build_output_path(
    default_stem: str, override: str | None, default_ext: str, publish: bool
) -> Path:
    """Resolve output path and enforce EPS extension in publish mode."""
    if override:
        out_path = Path(override)
    else:
        out_path = Path(f"{default_stem}.{default_ext}")

    if publish and out_path.suffix.lower() != ".eps":
        out_path = out_path.with_suffix(".eps")
    return out_path


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
        help=(
            "Output filename for Gantt chart "
            "(default: <input_file>_gantt.png or .eps with --publish)"
        ),
    )
    parser.add_argument(
        "--production-output",
        default=None,
        help=(
            "Output filename for production bar chart "
            "(default: <input_file>_production.eps with --publish; ignored otherwise)"
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publication mode: save figures as EPS files.",
    )
    parser.add_argument(
        "--stn",
        action="store_true",
        help="Also generate a State-Task Network graph from TOML data.",
    )
    parser.add_argument(
        "--toml",
        default=None,
        help=(
            "Path to TOML file used to build the STN graph "
            "(default: auto from results/model name)"
        ),
    )
    parser.add_argument(
        "--stn-output",
        default=None,
        help=(
            "Output filename for STN graph "
            "(default: <toml_file>_stn.png or .eps with --publish)"
        ),
    )
    parser.add_argument(
        "--hide-dsch",
        action="store_true",
        help="Hide dsch from plots (production bar chart and STN dsch arrows).",
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
    if data.get("revenue") is not None:
        print(f"  Revenue  : {data.get('revenue')}")

    model_name = str(data.get("model_name", ""))
    toml_path = Path(args.toml) if args.toml else infer_toml_path(filepath, model_name)
    if not args.toml:
        print(f"  TOML     : auto -> {toml_path}")
    else:
        print(f"  TOML     : {toml_path}")

    deadline_hours = None
    if is_economic_context(filepath, model_name):
        deadline_hours = read_deadline_hours(toml_path)
        if deadline_hours is not None:
            print(f"  Deadline : {deadline_hours} h")

    has_production = bool(data.get("production"))

    if args.publish:
        fig_main = plt.figure(figsize=(16, 8), layout="constrained")
        ax_gantt = fig_main.add_subplot(111)

        if has_production:
            fig_prod = plt.figure(figsize=(16, 5), layout="constrained")
            ax_prod = fig_prod.add_subplot(111)
        else:
            fig_prod = None
            ax_prod = None
    else:
        fig_main = plt.figure(figsize=(16, 10), layout="constrained")
        if has_production:
            gs = fig_main.add_gridspec(2, 1, height_ratios=[3, 1])
            ax_gantt = fig_main.add_subplot(gs[0])
            ax_prod = fig_main.add_subplot(gs[1])
            fig_prod = None
        else:
            ax_gantt = fig_main.add_subplot(111)
            ax_prod = None
            fig_prod = None

    if has_production and args.production_output is not None and not args.publish:
        print("Note: --production-output is ignored unless --publish is set.")

    plot_solution(
        data, ax_gantt, deadline_hours=deadline_hours, publish_mode=args.publish
    )
    if ax_prod is not None:
        plot_production(data, ax_prod, show_dsch=not args.hide_dsch)

    if data.get("generated") and not args.publish:
        fig_main.text(
            0.99,
            0.012,
            f"Generated: {data['generated']}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
        )
        if fig_prod is not None:
            fig_prod.text(
                0.99,
                0.012,
                f"Generated: {data['generated']}",
                ha="right",
                va="bottom",
                fontsize=7,
                color="gray",
            )

    default_ext = "eps" if args.publish else "png"
    save_main_requested = (
        args.no_show
        or args.output is not None
        or args.publish
        or args.production_output is not None
    )

    if save_main_requested:
        if args.publish:
            gantt_out = build_output_path(
                f"{filepath.stem}_gantt", args.output, default_ext, args.publish
            )
            fig_main.savefig(gantt_out, dpi=800, bbox_inches="tight")
            print(f"Saved Gantt chart to {gantt_out}")

            if fig_prod is not None:
                prod_out = build_output_path(
                    f"{filepath.stem}_production",
                    args.production_output,
                    default_ext,
                    args.publish,
                )
                fig_prod.savefig(prod_out, dpi=800, bbox_inches="tight")
                print(f"Saved production chart to {prod_out}")
        else:
            prod_out = build_output_path(
                f"{filepath.stem}_gantt",
                args.output,
                default_ext,
                args.publish,
            )
            fig_main.savefig(prod_out, dpi=800, bbox_inches="tight")
            print(f"Saved combined chart to {prod_out}")

    if args.stn:
        if not args.toml:
            print(f"Auto-selected STN TOML: {toml_path}")
        if not toml_path.exists():
            print(f"Error: TOML file '{toml_path}' not found. STN graph skipped.")
        else:
            fig_stn = plt.figure(figsize=(18, 10), layout="constrained")
            ax_stn = fig_stn.add_subplot(111)
            plot_stn_graph(str(toml_path), ax_stn, show_dsch=not args.hide_dsch)
            if args.no_show or args.stn_output or args.publish:
                stn_out = build_output_path(
                    f"{toml_path.stem}_stn", args.stn_output, default_ext, args.publish
                )
                fig_stn.savefig(stn_out, dpi=450, bbox_inches="tight")
                print(f"Saved STN graph to {stn_out}")

    if not (
        save_main_requested
        or args.stn_output is not None
        or (args.publish and args.stn)
    ):
        plt.show()


if __name__ == "__main__":
    main()
