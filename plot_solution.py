"""
Plot solution from a results .txt file.

Usage:
    python plot_solution.py                 # plots results.txt
    python plot_solution.py results_gdp.txt # plots a specific file
    python plot_solution.py --no-show       # save to PNG without showing
    python plot_solution.py --stn --no-show # also plot the STN graph
    python plot_solution.py --publish       # save to PDF (publication mode)

Produces two figures:
    1. Gantt chart of the task schedule (one bar per unit)
    2. Bar chart of final production vs. demand per product
"""

import argparse
import math
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

TASK_COLORS: dict[str, str] = {
    "Pr": "#C06C84",  # pressing
    "Fa": "#E07B54",  # alcoholic fermentation
    "Fl": "#5B8DB8",  # malolactic fermentation
    "Age": "#C8A97E",  # aging (barrel / jar)
    "Cs": "#6BBF7A",  # cold stabilization
    "Stg": "#9B7EC8",  # storage
}

# Unit order for the y-axis (makespan models)
UNIT_ORDER_MS: list[str] = [
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

# Unit order for the y-axis (economic model)
UNIT_ORDER_ECONOMIC: list[str] = [
    "press",
    "inox_ext_5000v#1",
    "inox_ext_10000#1",
    "inox_ext_10000#2",
    "subte_6200",
    "subte_9500",
    "iso_ext_5000",
    "iso_ext_10000",
    "barrique [1]",
    "barrique [2]",
    "barrique [3]",
    "jar [1]",
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
    "inox_ext_5000v#1": "Inox tanks",
    "inox_ext_10000#1": "Inox tanks",
    "inox_ext_10000#2": "Inox tanks",
    "subte_6200": "Subterranean tanks",
    "subte_9500": "Subterranean tanks",
    "iso_ext_5000": "Isothermal tanks",
    "iso_ext_10000": "Isothermal tanks",
    "barrique [1]": "Barrique pool",
    "barrique [2]": "Barrique pool",
    "barrique [3]": "Barrique pool",
    "jar [1]": "Jar pool",
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

    for t in tasks:
        for u in t["units"]:
            m = re.match(r"^(barrique|jar)x(\d+)$", u["unit"])
            if m:
                count = int(m.group(2))
                u["unit"] = m.group(1)
                u["pool_count"] = count
                u["batch"] = round(u["batch"] * count, 2)

    pool_lane_counts: dict[str, int] = {}

    def assign_pool_lanes(tasks: list[dict], pool: str) -> None:
        pool_entries = [(t, u) for t in tasks for u in t["units"] if u["unit"] == pool]
        pool_entries.sort(key=lambda x: x[0]["start"])
        lane_ends: list[float] = []
        for t, u in pool_entries:
            lane = next(
                (i for i, end in enumerate(lane_ends) if t["start"] >= end - 1e-6),
                len(lane_ends),
            )
            if lane == len(lane_ends):
                lane_ends.append(t["end"])
            else:
                lane_ends[lane] = t["end"]
            lane_name = f"{pool} [{lane + 1}]"
            pool_count = int(u.get("pool_count", 0) or 0)
            if pool_count > 0:
                pool_lane_counts[lane_name] = max(
                    pool_lane_counts.get(lane_name, 0), pool_count
                )
            u["unit"] = lane_name

    assign_pool_lanes(tasks, "barrique")
    assign_pool_lanes(tasks, "jar")

    meta["tasks"] = tasks
    meta["pool_lane_counts"] = pool_lane_counts
    meta["wine_metadata"] = parse_wine_metadata(text)
    meta["production"] = parse_production(text)
    discard_map = parse_discard_data(text)
    for entry in meta["production"]:
        entry.setdefault("discarded", discard_map.get(entry["product"], 0.0))
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

    meta["profit"] = parse_metric(r"Profit\s*[:$\s]*([-\d.,]+)")
    meta["revenue"] = parse_metric(r"Revenue\s*[:$\s]*([\d.,]+)")
    meta["grape_skin_revenue"] = parse_metric(r"Grape Skin Revenue\s*[:$\s]*([\d.,]+)")
    meta["outsourcing_cost"] = parse_metric(r"Outsourcing Cost\s*[:$\s]*([\d.,]+)")
    meta["raw_material_cost"] = parse_metric(r"Raw Material Cost\s*[:$\s]*([\d.,]+)")
    meta["lateness_cost"] = parse_metric(r"Lateness Cost\s*[:$\s]*([\d.,]+)")
    meta["penalty_empty_tank"] = parse_metric(r"Penalty Empty Tank\s*[:$\s]*([\d.,]+)")
    meta["penalty_air_space"] = parse_metric(r"Penalty Air Space\s*[:$\s]*([\d.,]+)")
    meta["penalty_makespan"] = parse_metric(r"Penalty Makespan\s*[:$\s]*([\d.,]+)")
    meta["cooling_cost"] = parse_metric(r"Cooling Cost\s*[:$\s]*([\d.,]+)")

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


def parse_discard_data(text: str) -> dict[str, float]:
    discard_map: dict[str, float] = {}
    section = re.search(r"DISCARD DATA.*?\n[-]+\n(.*?)(?:\n={5,}|\Z)", text, re.DOTALL)
    if not section:
        return discard_map
    for line in section.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            try:
                discard_map[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                pass
    return discard_map


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


def read_deadlines(toml_path: Path) -> list[float]:
    """Return a list of deadline values (hours) for all campaigns, or [] if unavailable."""
    if not toml_path.exists():
        return []
    with open(toml_path, "rb") as f:
        params = tomllib.load(f)
    global_cfg = params.get("global", {})

    if "deadlines" in global_cfg:
        try:
            return [float(d) for d in global_cfg["deadlines"]]
        except (TypeError, ValueError):
            return []


def task_stage(task_name: str) -> str:
    return task_name.rstrip("0123456789")


def task_line(task_name: str) -> str:
    i = len(task_name)
    while i > 0 and task_name[i - 1].isdigit():
        i -= 1
    return task_name[i:]


def task_stage_key(task_name: str) -> str:
    """Return the canonical stage family used for plotting and coloring."""
    stage = task_stage(task_name)
    if stage.startswith("Age"):
        return "Age"
    return stage


def format_task_label(task_name: str, publish_mode: bool) -> str:
    """Return a display label for task codes, expanded in publish mode."""
    if not publish_mode:
        return task_name

    stage = task_stage(task_name)
    line = task_line(task_name)
    static_names = {
        "Fa": "Alcoholic Fermentation",
        "Fl": "Malolactic Fermentation",
        "Stg": "Storage",
        "Cs": "Cold Stabilization",
    }
    if stage in static_names and line:
        return f"{static_names[stage]} ({line})"
    m_age = re.match(r"^Age(Bar|Jar)(\d+)M$", stage)
    if m_age and line:
        vessel = "Barrique" if m_age.group(1) == "Bar" else "Jar"
        months = m_age.group(2)
        return f"{vessel} Aging ({months} months) ({line})"
    return task_name


def format_unit_label(
    unit_name: str, publish_mode: bool, pool_count: int | None = None
) -> str:
    """Return a display label for units, expanded in publish mode."""
    UNIT_PUBLISH_NAMES: dict[str, str] = {
        "press": "Press",
        "barrique": "Barrique Pool",
        "jar": "Jar Pool",
    }
    if unit_name in UNIT_PUBLISH_NAMES:
        return UNIT_PUBLISH_NAMES[unit_name]

    m = re.match(r"^(inox|subte|iso)(\d+)$", unit_name, re.IGNORECASE)
    if m:
        family = m.group(1).lower().capitalize()
        capacity = m.group(2)
        return f"{family} {capacity}"

    m = re.match(
        r"^(inox|subte|iso)_ext_(\d+)(v)?(?:#(\d+))?$", unit_name, re.IGNORECASE
    )
    if m:
        family_raw = m.group(1).lower()
        family = {
            "inox": "Inox.",
            "subte": "Subte.",
            "iso": "Iso.",
        }.get(family_raw, family_raw.capitalize())
        capacity = m.group(2)
        is_variable = bool(m.group(3))
        idx = m.group(4)
        cap_label = f"{capacity}V" if is_variable else capacity
        return f"{family} {cap_label} #{idx}" if idx else f"{family} {cap_label}"

    m = re.match(r"^subte_(\d+)$", unit_name, re.IGNORECASE)
    if m:
        return f"Subte. {m.group(1)}"

    m = re.match(r"^(barrique|jar)\s*\[(\d+)\]$", unit_name, re.IGNORECASE)
    if m:
        family = "Barrique" if m.group(1).lower() == "barrique" else "Jar"
        lane = m.group(2)
        count_suffix = f" (x{pool_count})" if pool_count and pool_count > 0 else ""
        if publish_mode:
            return f"{family} pool {lane}{count_suffix}"
        return f"{family} {lane}{count_suffix}"

    m = re.match(r"^(.+)#(\d+)$", unit_name)
    if m:
        family = m.group(1).capitalize()
        number = m.group(2)
        return f"{family} #{number}"

    return unit_name


def parse_unit_family(unit_name: str) -> tuple[str, int | None]:
    """Return (family, number) for units like 'jar#5', else (unit_name, None)."""
    m = re.match(r"^(.+)#(\d+)$", unit_name)
    if m:
        return m.group(1), int(m.group(2))
    return unit_name, None


def build_collapsed_unit_groups(
    tasks: list[dict],
) -> tuple[dict[str, str], dict[str, tuple[str, str, int]]]:
    """
    Find units with the same family name (e.g. 'jar', 'barrique') and identical
    timeline (same set of (start, end) intervals across all tasks) and group them.

    Returns:
        unit_to_rep  : maps each unit name to the group representative unit name.
        rep_to_info  : maps representative -> (first_unit, last_unit, count).
    """
    unit_intervals: dict[str, list[tuple[str, float, float]]] = {}
    for t in tasks:
        for u in t["units"]:
            name = u["unit"]
            interval = (t["task"], t["start"], t["end"])
            if name not in unit_intervals:
                unit_intervals[name] = []
            if interval not in unit_intervals[name]:
                unit_intervals[name].append(interval)

    family_groups: dict[str, dict[frozenset, list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for name, intervals in unit_intervals.items():
        family, number = parse_unit_family(name)
        if number is not None:
            key = frozenset(intervals)
            family_groups[family][key].append((number, name))

    unit_to_rep: dict[str, str] = {}
    rep_to_info: dict[str, tuple[str, str, int]] = {}

    for _family, by_intervals in family_groups.items():
        for _key, members in by_intervals.items():
            if len(members) < 2:
                continue
            members_sorted = sorted(members, key=lambda x: x[0])
            rep = members_sorted[0][1]
            first_unit = members_sorted[0][1]
            last_unit = members_sorted[-1][1]
            rep_to_info[rep] = (first_unit, last_unit, len(members))
            for _num, name in members:
                unit_to_rep[name] = rep

    return unit_to_rep, rep_to_info


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
    shared_tasks_added: set[str] = set()
    for line, cfg in lines_cfg.items():
        line_product[line] = cfg["product"]
        for step in cfg["steps"]:
            if step == "Pr":
                if "Pr" not in shared_tasks_added:
                    tasks.append("Pr")
                    shared_tasks_added.add("Pr")
            else:
                tasks.append(f"{step}{line}")

    states: set[str] = {"dsch"}
    for line, cfg in lines_cfg.items():
        if "Pr" in cfg["steps"]:
            states.add("m")
        states.add(f"v{line}")
        if "Fl" in cfg["steps"]:
            states.add(f"vl{line}")
        states.add(cfg["product"])

    # Include any explicitly declared inventory states. This is important when
    # raw materials are shared pools (e.g. s_red, s_white_rose) instead of s1..sN.
    states.update(initial_inventory.keys())

    # Detect stg lines. vbuf and AgeViaStg are internal model details; the STN shows
    # Stg as a single storage circle with arcs vl→Stg and Stg→va.
    stg_lines: list[str] = []
    for line, cfg in lines_cfg.items():
        if "Stg" in cfg["steps"]:
            stg_lines.append(line)

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

    # Synthesise arcs for Stg (rho not in TOML).
    # Shown as two arcs: vl→Stg (consume) and Stg→Age{line} (produce), collapsing
    # the internal vbuf/AgeViaStg chain. Wine held in Stg feeds the same aging task
    # (same barriques/jars) as the direct path, Stg is a pre-aging hold only.
    for line, cfg in lines_cfg.items():
        if "Stg" not in cfg["steps"]:
            continue
        aging_step = next((s for s in cfg["steps"] if s.startswith("Age")), None)
        if not aging_step:
            continue
        has_fl = "Fl" in cfg["steps"]
        pre_ag = f"vl{line}" if has_fl else f"v{line}"
        stg_task = f"Stg{line}"
        aging_task = f"{aging_step}{line}"
        arcs += [
            {"src": pre_ag, "dst": stg_task, "coef": 1.0, "kind": "cons"},
            {"src": stg_task, "dst": aging_task, "coef": 1.0, "kind": "prod"},
        ]

    return {
        "tasks": tasks,
        "states": sorted(states),
        "arcs": arcs,
        "lines_cfg": lines_cfg,
        "line_product": line_product,
        "stg_lines": stg_lines,
        "products": products,
        "initial_inventory": initial_inventory,
        "product_ub": product_ub,
    }


def format_arc_label(value: float) -> str:
    if 0 <= value <= 1.5:
        return f"{value * 100:.0f}%"
    return f"{value:.2f}"


# Y-offset for the Stg detour path relative to the line's main row.
_STG_Y_OFFSET = -0.35


def build_stn_positions(stn_data: dict) -> dict[str, tuple[float, float]]:
    lines_cfg = stn_data["lines_cfg"]
    tasks = stn_data["tasks"]
    states = stn_data["states"]
    line_product = stn_data["line_product"]
    arcs = stn_data["arcs"]
    stg_lines: set[str] = set(stn_data.get("stg_lines", []))

    sorted_lines = sorted(lines_cfg.keys(), key=lambda x: int(x))
    y_by_line = {line: -idx for idx, line in enumerate(sorted_lines)}

    stage_x = {
        "Pr": 1.3,
        "Fa": 2.3,
        "Fl": 3.4,
        "Age": 4.1,
        "Cs": 5.0,
        # Stg and AgeViaStg are positioned per-task below using _STG_Y_OFFSET
    }

    # Place material states between the stages that consume/produce them to preserve
    # the visual causal direction (e.g.: s -> Pr -> m -> Fa).
    state_x = {
        "s": 0.2,
        "m": 1.8,
        "v": 2.95,
        "vl": 3.8,
        "va": 4.45,
        "p": 7.0,
    }

    pos: dict[str, tuple[float, float]] = {}

    for task in tasks:
        stage = task_stage_key(task)
        if task == "Pr":
            pr_ys = [
                y_by_line[l] for l in sorted_lines if "Pr" in lines_cfg[l]["steps"]
            ]
            y_pr = sum(pr_ys) / len(pr_ys) if pr_ys else 0.0
            pos["Pr"] = (stage_x.get("Pr", 1.3), y_pr)
        elif stage == "Stg":
            # Stg holding tank: between Fl and Age, on the Stg detour row
            line = task_line(task)
            pos[task] = (3.85, y_by_line[line] + _STG_Y_OFFSET)
        else:
            line = task_line(task)
            pos[task] = (stage_x.get(stage, 3.0), y_by_line[line])

    task_line_map = {task: task_line(task) for task in tasks}

    pressing_lines = [ln for ln in sorted_lines if "Pr" in lines_cfg[ln]["steps"]]

    def state_connected_lines(state: str) -> list[str]:
        connected: list[str] = []
        for arc in arcs:
            if arc["src"] == state and arc["dst"] in task_line_map:
                line = task_line_map[arc["dst"]]
                if line in y_by_line:
                    connected.append(line)
                elif arc["dst"] == "Pr":
                    connected.extend(pressing_lines)
            elif arc["dst"] == state and arc["src"] in task_line_map:
                line = task_line_map[arc["src"]]
                if line in y_by_line:
                    connected.append(line)
                elif arc["src"] == "Pr":
                    connected.extend(pressing_lines)
        return connected

    for state in states:
        if state == "dsch":
            pos[state] = (6.8, 1.2)
            continue
        if state == "grape_skin":
            pos[state] = (state_x["s"], -(len(sorted_lines) - 1) - 0.7)
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
        elif state.startswith("va"):
            pos[state] = (state_x["va"], y)
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
    stg_task_set = {f"Stg{ln}" for ln in stn_data.get("stg_lines", [])}
    for task in task_nodes:
        x, y = pos[task]
        stage = task_stage_key(task)
        if task in stg_task_set:
            # Stg: storage circle, visually distinct from processing triangles
            ax.scatter(
                [x],
                [y],
                s=280,
                marker="o",
                c=TASK_COLORS.get("Stg", "#9B7EC8"),
                edgecolors="#4A235A",
                linewidths=1.5,
                zorder=4,
            )
            ax.text(x, y - 0.32, task, fontsize=7, ha="center", va="top", zorder=5)
        else:
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
    has_stg = bool(stn_data.get("stg_lines"))
    ax.set_ylim(-len(stn_data["lines_cfg"]) - (1.2 if has_stg else 0.8), 2.1)
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
    ax.legend(handles=legend_items, loc="upper left", fontsize=7, framealpha=0.9)


def compute_xaxis_breaks(
    tasks: list[dict],
    xmax: float,
    min_gap_fraction: float = 0.08,
    max_breaks: int | None = 2,
) -> list[tuple[float, float]]:
    """
    Return (break_start, break_end) intervals where no non-aging task is active
    and the gap exceeds min_gap_fraction * xmax (minimum 500 h).
    If max_breaks is not None, keep only the largest max_breaks intervals.
    Only tasks whose name starts with 'Age' are treated as aging tasks.
    """
    min_gap = max(500.0, xmax * min_gap_fraction)

    events: list[tuple[float, int]] = []
    for t in tasks:
        m = re.match(r"[A-Za-z]+", t["task"])
        prefix = m.group() if m else ""
        if not prefix.startswith("Age"):
            events.append((t["start"], +1))
            events.append((t["end"], -1))

    if not events:
        return []

    events.sort()
    active = 0
    inactive_start: float | None = None
    candidates: list[tuple[float, float, float]] = []

    for time, delta in events:
        prev = active
        active += delta
        if prev > 0 and active == 0:
            inactive_start = time
        elif prev == 0 and active > 0 and inactive_start is not None:
            gap = time - inactive_start
            if gap >= min_gap:
                pad = min(gap * 0.03, 48.0)
                candidates.append((inactive_start + pad, time - pad, gap))
            inactive_start = None

    if not candidates:
        return []

    if max_breaks is not None and max_breaks >= 0 and len(candidates) > max_breaks:
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:max_breaks]
        candidates.sort(key=lambda x: x[0])

    breaks = [(b0, b1) for b0, b1, _ in candidates]
    return breaks


def _add_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """
    Draw diagonal break marks at the boundary between two adjacent axes.
    """
    d = 0.4
    marker = [(-1, -d), (1, d)]
    kw = dict(
        marker=marker,
        markersize=9,
        linestyle="none",
        color="k",
        mec="k",
        mew=1.2,
        clip_on=False,
        zorder=10,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kw)  # type: ignore[arg-type]
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kw)  # type: ignore[arg-type]


DEADLINE_COLORS = ["#B03A2E", "#2471A3", "#1E8449", "#7D3C98", "#D68910", "#0E6655"]


def plot_solution(
    data: dict,
    ax: plt.Axes,
    deadlines: list[float] | None = None,
    publish_mode: bool = False,
    no_title: bool = False,
    no_break: bool = False,
    max_breaks: int | None = 2,
) -> None:
    tasks = data["tasks"]
    model_name_l = str(data.get("model_name", "")).lower()
    is_economic = "economic" in model_name_l

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
    pool_lane_counts = data.get("pool_lane_counts", {})
    for t in tasks:
        for u in t["units"]:
            if u["unit"] not in used_units:
                used_units.append(u["unit"])

    unit_to_rep, rep_to_info = build_collapsed_unit_groups(tasks)

    unit_order = UNIT_ORDER_ECONOMIC if is_economic else UNIT_ORDER_MS
    ordered_units = [u for u in unit_order if u in used_units]
    ordered_units += [u for u in used_units if u not in ordered_units]

    seen_reps: set[str] = set()
    collapsed_ordered: list[str] = []
    for u in ordered_units:
        rep = unit_to_rep.get(u, u)
        if rep not in seen_reps:
            seen_reps.add(rep)
            collapsed_ordered.append(rep)
    ordered_units = collapsed_ordered

    unit_index = {u: i for i, u in enumerate(ordered_units)}
    bar_height = 0.6

    xmax = max(t["end"] for t in tasks) * 1.01

    # Build segments from x-axis breaks
    breaks = (
        [] if no_break else compute_xaxis_breaks(tasks, xmax, max_breaks=max_breaks)
    )
    fig = ax.get_figure()
    assert fig is not None

    if breaks:
        ss = ax.get_subplotspec()
        assert ss is not None
        ax.remove()

        segments: list[tuple[float, float]] = []
        prev = 0.0
        for b_start, b_end in breaks:
            segments.append((prev, b_start))
            prev = b_end
        segments.append((prev, xmax))

        def segment_width(duration: float) -> float:
            cap = 2400.0
            return max(cap * (1.0 - math.exp(-duration / cap)), 1.0)

        widths = [segment_width(seg_end - seg_start) for seg_start, seg_end in segments]
        gss = GridSpecFromSubplotSpec(
            1,
            len(segments),
            subplot_spec=ss,
            wspace=0.04,
            width_ratios=widths,
        )
        axes = [fig.add_subplot(gss[0, i]) for i in range(len(segments))]
    else:
        axes = [ax]
        segments = [(0.0, xmax)]

    best_seg: dict[tuple[str, int, str], int] = {}
    best_seg_width: dict[tuple[str, int, str], float] = {}
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        for t in tasks:
            clip_w = min(t["end"], seg_end) - max(t["start"], seg_start)
            if clip_w <= 0:
                continue
            for u in t["units"]:
                display_unit = unit_to_rep.get(u["unit"], u["unit"])
                key = (t["task"], t["event"], display_unit)
                if clip_w > best_seg_width.get(key, 0):
                    best_seg[key] = seg_idx
                    best_seg_width[key] = clip_w

    def bar_size_px(
        bar_ax: plt.Axes,
        left: float,
        right: float,
        y_center: float,
        height: float,
    ) -> tuple[float, float]:
        """Return (width_px, height_px) for a bar segment in display pixels."""
        p0 = bar_ax.transData.transform((left, y_center - height / 2))
        p1 = bar_ax.transData.transform((right, y_center + height / 2))
        return abs(float(p1[0] - p0[0])), abs(float(p1[1] - p0[1]))

    def text_box_px(text: str, fontsize: float, dpi: float) -> tuple[float, float]:
        """Estimate text bounding box size in pixels for conservative fit checks."""
        lines = text.split("\n")
        max_chars = max((len(line) for line in lines), default=0)
        char_w_px = (fontsize * dpi / 72.0) * 0.66
        line_h_px = (fontsize * dpi / 72.0) * 1.30
        width_px = max_chars * char_w_px + 8.0
        height_px = len(lines) * line_h_px + 4.0
        return width_px, height_px

    def fits_in_bar(
        text: str,
        fontsize: float,
        bar_w_px: float,
        bar_h_px: float,
        dpi: float,
    ) -> bool:
        text_w_px, text_h_px = text_box_px(text, fontsize, dpi)
        return text_w_px <= bar_w_px * 0.82 and text_h_px <= bar_h_px * 0.78

    for seg_idx, (seg_ax, (seg_start, seg_end)) in enumerate(zip(axes, segments)):
        for t in tasks:
            m = re.match(r"[A-Za-z]+", t["task"])
            prefix = m.group() if m else ""
            if prefix in TASK_COLORS:
                color = TASK_COLORS[prefix]
            elif prefix.startswith("Age"):
                color = TASK_COLORS["Age"]
            else:
                color = "#AAAAAA"

            duration = t["end"] - t["start"]
            clip_start = max(t["start"], seg_start)
            clip_end = min(t["end"], seg_end)
            if clip_end <= clip_start:
                continue

            _age_months_m = re.search(r"Age(?:Bar|Jar)(\d+)M", t["task"])
            is_long_aging = bool(_age_months_m) and int(_age_months_m.group(1)) > 5
            hatch = "////" if is_long_aging else None
            edge_color = "#4B2E1E" if is_long_aging else "#52514e"
            bar_gap = min(6.0, max(1.0, (clip_end - clip_start) * 0.03))
            draw_left = clip_start + bar_gap / 2.0
            draw_width = max(clip_end - clip_start - bar_gap, 0.5)

            drawn_display_units: set[str] = set()
            for u in t["units"]:
                display_unit = unit_to_rep.get(u["unit"], u["unit"])
                yi = unit_index.get(display_unit)
                if yi is None or display_unit in drawn_display_units:
                    continue
                drawn_display_units.add(display_unit)

                seg_ax.barh(
                    yi,
                    draw_width,
                    left=draw_left,
                    height=bar_height,
                    color=color,
                    edgecolor=edge_color,
                    linewidth=0.9,
                    align="center",
                    hatch=hatch,
                )

                # Label only in the segment with the largest visible clip
                label_key = (t["task"], t["event"], display_unit)
                if best_seg.get(label_key) == seg_idx:
                    bar_w_px, bar_h_px = bar_size_px(
                        seg_ax, draw_left, draw_left + draw_width, yi, bar_height
                    )
                    clipped_width = draw_width

                    label_x = t["start"] + duration / 2
                    if not (seg_start <= label_x <= seg_end):
                        label_x = draw_left + draw_width / 2
                    display_task = format_task_label(t["task"], publish_mode)
                    batch_str = (
                        f"{t['batch']:.0f} L"
                        if display_unit in rep_to_info
                        else f"{u['batch']:.0f} L"
                    )

                    # Aggressive filtering: label only major bars to avoid clutter.
                    if clipped_width < 720 or bar_w_px < 260 or bar_h_px < 20:
                        continue

                    dpi = float(seg_ax.figure.dpi)
                    two_line_text = f"{display_task}\n{batch_str}"
                    if fits_in_bar(two_line_text, 6.5, bar_w_px, bar_h_px, dpi):
                        label_text = two_line_text
                        label_fs = 6.5
                    else:
                        continue

                    seg_ax.text(
                        label_x,
                        yi,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=label_fs,
                        color="white",
                        fontweight="bold",
                        clip_on=True,
                    )

    def make_ytick_label(u: str) -> str:
        if u in rep_to_info:
            first_unit, last_unit, count = rep_to_info[u]
            if publish_mode:
                first_label = format_unit_label(
                    first_unit,
                    publish_mode,
                    pool_lane_counts.get(first_unit),
                )
                last_label = format_unit_label(
                    last_unit,
                    publish_mode,
                    pool_lane_counts.get(last_unit),
                )
            else:
                first_label, last_label = first_unit, last_unit
            return f"{first_label} - {last_label} (x{count})"
        return format_unit_label(u, publish_mode, pool_lane_counts.get(u))

    n_units = len(ordered_units)
    display_units = [make_ytick_label(u) for u in ordered_units]

    for i, seg_ax in enumerate(axes):
        seg_ax.set_yticks(range(n_units))
        seg_ax.set_ylim(-0.5, n_units - 0.5)
        seg_ax.invert_yaxis()
        seg_ax.set_axisbelow(True)
        if i == 0:
            seg_ax.set_yticklabels(display_units, fontsize=9)
            seg_ax.set_ylabel("Unit", fontsize=10)
        else:
            seg_ax.set_yticklabels([])
            seg_ax.tick_params(axis="y", length=0)
            seg_ax.spines["left"].set_visible(False)

    def pick_hour_interval(width_h: float) -> tuple[int, int]:
        """Return (major, minor) hour tick intervals for a segment of given width."""
        for major, minor in [(24, 6), (168, 24), (504, 168), (1008, 168), (2016, 336)]:
            if is_economic:
                if width_h / major <= 10:
                    return major, minor
            else:
                if width_h / major <= 15:
                    return major, minor
        return 4032, 672

    def pick_day_interval(width_d: float) -> int:
        for interval in [7, 14, 28, 56, 91, 182]:
            if is_economic:
                if width_d / interval <= 10:
                    return interval
            else:
                if width_d / interval <= 15:
                    return interval
        return 365

    for i, (seg_ax, (seg_start, seg_end)) in enumerate(zip(axes, segments)):
        seg_ax.set_xlim(seg_start, seg_end)
        if i < len(axes) - 1:
            seg_ax.spines["right"].set_visible(False)

        if is_economic:
            # Days-only bottom axis
            major_d = pick_day_interval((seg_end - seg_start) / 24)
            seg_ax.xaxis.set_major_locator(MultipleLocator(major_d * 24))
            seg_ax.xaxis.set_minor_locator(MultipleLocator(major_d * 24 / 4))
            seg_ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x / 24:.0f}")
            )
        else:
            major_h, minor_h = pick_hour_interval(seg_end - seg_start)
            seg_ax.xaxis.set_minor_locator(MultipleLocator(minor_h))
            seg_ax.xaxis.set_major_locator(MultipleLocator(major_h))

        seg_ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        seg_ax.grid(axis="x", which="major", linestyle="--", linewidth=0.6, alpha=0.5)

        if not is_economic:
            ax2 = seg_ax.twiny()
            ax2.set_xlim(seg_start / 24, seg_end / 24)
            major_d = pick_day_interval((seg_end - seg_start) / 24)
            ax2.xaxis.set_major_locator(MultipleLocator(major_d))
            if i < len(axes) - 1:
                ax2.spines["right"].set_visible(False)
            if i > 0:
                ax2.spines["left"].set_visible(False)

    # Place x-axis labels centred across all segments
    if is_economic:
        if len(axes) == 1:
            axes[0].set_xlabel("Time (days)", fontsize=10)
        else:
            day_label = fig.text(
                0.55,
                -0.025,
                "Time (days)",
                ha="center",
                va="bottom",
                fontsize=10,
            )
            day_label.set_in_layout(True)
    else:
        if len(axes) == 1:
            axes[0].set_xlabel("Time (hours)", fontsize=10)
            ax2.set_xlabel("Time (days)", fontsize=10)
        else:
            ann_b = axes[0].annotate(
                "Time (hours)",
                xy=(0.5, -0.03),
                xycoords=("figure fraction", "axes fraction"),
                ha="center",
                va="top",
                fontsize=10,
                annotation_clip=False,
            )
            ann_b.set_in_layout(False)
            ann_t = axes[0].annotate(
                "Time (days)",
                xy=(0.5, 1.07),
                xycoords=("figure fraction", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=10,
                annotation_clip=False,
            )
            ann_t.set_in_layout(False)

    for dl_idx, dl_hours in enumerate(deadlines or []):
        dl_color = DEADLINE_COLORS[dl_idx % len(DEADLINE_COLORS)]
        for seg_ax, (seg_start, seg_end) in zip(axes, segments):
            if seg_start <= dl_hours <= seg_end:
                seg_ax.axvline(
                    dl_hours,
                    color=dl_color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=0,
                )

    for i in range(len(axes) - 1):
        _add_break_marks(axes[i], axes[i + 1])

    legend_defs = [
        ("Pr", "Pressing", TASK_COLORS["Pr"]),
        ("Fa", "Alcoholic fermentation", TASK_COLORS["Fa"]),
        ("Fl", "Malolactic fermentation", TASK_COLORS["Fl"]),
        ("Age", "Aging", TASK_COLORS["Age"]),
        ("Cs", "Cold stabilization", TASK_COLORS["Cs"]),
        ("Stg", "Storage", TASK_COLORS["Stg"]),
    ]
    legend_handles: list[Artist] = [
        mpatches.Patch(
            facecolor=color,
            label=name if publish_mode else f"{name} ({code})",
        )
        for code, name, color in legend_defs
        if any(t["task"].startswith(code) for t in tasks)
    ]
    for dl_idx, dl_hours in enumerate(deadlines or []):
        dl_color = DEADLINE_COLORS[dl_idx % len(DEADLINE_COLORS)]
        _dl_name = f"Deadline {dl_idx + 1}"
        _dl_label = (
            f"{_dl_name} ({dl_hours / 24:.0f} d)"
            if is_economic
            else f"{_dl_name} ({dl_hours:.0f} h)"
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=dl_color,
                linestyle="--",
                linewidth=1.4,
                label=_dl_label,
            )
        )
    axes[-1].legend(
        handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85
    )

    title_parts = []
    if data.get("model_name"):
        title_parts.append(data["model_name"])
    if data.get("makespan"):
        title_parts.append(
            f"Makespan: {data['makespan']:.0f} h ({data['makespan'] / 24:.1f} days)"
        )
    model_name_l = str(data.get("model_name", "")).lower()
    if "economic" in model_name_l:
        if data.get("profit") is not None:
            title_parts.append(f"Profit: {data['profit']:.2f}")
        econ_parts = []
        if data.get("revenue") is not None:
            econ_parts.append(f"Rev: {data['revenue']:.2f}")
        if data.get("grape_skin_revenue") is not None:
            econ_parts.append(f"GS Rev: {data['grape_skin_revenue']:.2f}")
        if data.get("outsourcing_cost") is not None:
            econ_parts.append(f"Outsource: -{data['outsourcing_cost']:.2f}")
        if data.get("raw_material_cost") is not None:
            econ_parts.append(f"RawMat: -{data['raw_material_cost']:.2f}")
        if data.get("lateness_cost") is not None and data["lateness_cost"] > 0:
            econ_parts.append(f"Lateness: -{data['lateness_cost']:.2f}")
        if econ_parts:
            title_parts.append("; ".join(econ_parts))
        penalty_parts = []
        if data.get("penalty_empty_tank") is not None:
            penalty_parts.append(f"EmptyTank: -{data['penalty_empty_tank']:.2f}")
        if data.get("penalty_air_space") is not None:
            penalty_parts.append(f"AirSpace: -{data['penalty_air_space']:.2f}")
        if data.get("penalty_makespan") is not None:
            penalty_parts.append(f"MS: -{data['penalty_makespan']:.2f}")
        if penalty_parts:
            title_parts.append("Penalties: " + "; ".join(penalty_parts))
    else:
        if data.get("objective") is not None:
            title_parts.append(f"Obj: {data['objective']:.2f}")
        if data.get("revenue") is not None:
            title_parts.append(f"Revenue: {data['revenue']:.2f}")
    if data.get("unused_units") is not None:
        title_parts.append(f"Unused tanks: {data['unused_units']}")
    if not no_title and title_parts:
        title_str = "  |  ".join(title_parts)
        if len(axes) == 1:
            axes[0].set_title(title_str, fontsize=10, pad=18)
        else:
            fig.suptitle(title_str, fontsize=10)


# Production bar chart
def plot_production(data: dict, ax: plt.Axes, show_dsch: bool = True) -> None:
    production = data.get("production", [])

    byproduct_keys = {"dsch", "grape_skin"}
    main_prod = [p for p in production if p.get("product") not in byproduct_keys]
    byproducts: list[dict] = []
    if show_dsch:
        for _key in ("dsch", "grape_skin"):
            _entry = next((p for p in production if p.get("product") == _key), None)
            if _entry is not None:
                byproducts.append(_entry)
    production = main_prod + byproducts

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
    BYPRODUCT_LABELS: dict[str, str] = {
        "dsch": "Discharge\n(dsch)",
        "grape_skin": "Grape Skin\n(byproduct)",
    }
    products = []
    for p in production:
        product_key = p["product"]
        if product_key in BYPRODUCT_LABELS:
            products.append(BYPRODUCT_LABELS[product_key])
            continue
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
    discarded = [p.get("discarded", 0.0) for p in production]
    show_outsourced = any(v > 1e-9 for v in outsourced)
    show_discarded = any(v > 1e-9 for v in discarded)

    x = range(len(products))

    PROD_COLOR = "#5B8DB8"
    GS_COLOR = "#C8A97E"
    prod_colors = [
        GS_COLOR if p.get("product") == "grape_skin" else PROD_COLOR for p in production
    ]

    if show_outsourced and show_discarded:
        width = 0.14
        gap = 0.02
        delta = width + gap
        offsets = [-1.5 * delta, -0.5 * delta, 0.5 * delta, 1.5 * delta]
        ax.bar(
            [xi + offsets[0] for xi in x],
            outsourced,
            width,
            label="Outsourced",
            color="#E07B54",
            edgecolor="white",
            linewidth=0.6,
        )
        bars_prod = ax.bar(
            [xi + offsets[1] for xi in x],
            produced,
            width,
            label="Produced",
            color=prod_colors,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            [xi + offsets[2] for xi in x],
            discarded,
            width,
            label="Discarded",
            color="#C0392B",
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            [xi + offsets[3] for xi in x],
            demand,
            width,
            label="Demand",
            color="#AAAAAA",
            edgecolor="white",
            linewidth=0.6,
        )
    elif show_outsourced:
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
            color=prod_colors,
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
    elif show_discarded:
        width = 0.16
        gap = 0.02
        delta = width + gap
        bars_prod = ax.bar(
            [xi - delta for xi in x],
            produced,
            width,
            label="Produced",
            color=prod_colors,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            [xi for xi in x],
            discarded,
            width,
            label="Discarded",
            color="#C0392B",
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
            color=prod_colors,
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
    title_parts_prod = ["Final Production"]
    if show_outsourced:
        title_parts_prod.append("Outsourced")
    if show_discarded:
        title_parts_prod.append("Discarded")
    title_parts_prod.append("Demand")
    ax.set_title(" vs. ".join(title_parts_prod), fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    y_max = max(
        max(produced) if produced else 0.0,
        max(demand) if demand else 0.0,
        max(outsourced) if show_outsourced else 0.0,
        max(discarded) if show_discarded else 0.0,
    )
    ax.set_ylim(0, y_max * 1.18)

    gs_rev = data.get("grape_skin_revenue")
    gs_xi = next(
        (i for i, p in enumerate(production) if p.get("product") == "grape_skin"),
        None,
    )

    for bar_i, bar in enumerate(bars_prod):
        h = bar.get_height()
        if h <= 0:
            continue
        bar_cx = bar.get_x() + bar.get_width() / 2
        if bar_i == gs_xi and gs_rev is not None:
            ax.text(
                bar_cx,
                h + y_max * 0.01,
                f"{h:,.0f} L\n${gs_rev:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#7D6608",
                rotation=45,
            )
        else:
            ax.text(
                bar_cx,
                h + y_max * 0.01,
                f"{h:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )


def build_output_path(
    default_stem: str,
    override: str | None,
    default_ext: str,
    publish: bool,
    publish_ext: str = "pdf",
) -> Path:
    """Resolve output path and enforce publish extension in publish mode."""
    if override:
        out_path = Path(override)
    else:
        out_path = Path(f"{default_stem}.{default_ext}")

    expected_suffix = f".{publish_ext.lower()}"
    if publish and out_path.suffix.lower() != expected_suffix:
        out_path = out_path.with_suffix(expected_suffix)
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
            "(default: <input_file>_gantt.png or .pdf with --publish)"
        ),
    )
    parser.add_argument(
        "--production-output",
        default=None,
        help=(
            "Output filename for production bar chart "
            "(default: <input_file>_production.pdf with --publish; ignored otherwise)"
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publication mode: save figures as PDF files (or SVG with --svg).",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="With --publish, save figures as SVG instead of PDF.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the title from the Gantt chart (useful in publication mode).",
    )
    parser.add_argument(
        "--no-break",
        action="store_true",
        help="Disable X-axis breaks; show the full uncompressed timeline.",
    )
    parser.add_argument(
        "--max-breaks",
        type=int,
        default=2,
        help=(
            "Maximum number of X-axis break intervals to keep (largest gaps). "
            "Use 0 to disable breaks unless --no-break is set."
        ),
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
            "(default: <toml_file>_stn.png or .pdf with --publish)"
        ),
    )
    parser.add_argument(
        "--hide-dsch",
        action="store_true",
        help="Hide dsch from plots (production bar chart and STN dsch arrows).",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=18.0,
        help="Figure width in inches for Gantt and production plots (default: 18).",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help=(
            "Optional figure height in inches for the main Gantt figure "
            "(default: 8 in publish mode, 10 otherwise)."
        ),
    )
    parser.add_argument(
        "--prod-height",
        type=float,
        default=5.0,
        help="Figure height in inches for production-only publish figure (default: 5).",
    )
    parser.add_argument(
        "--stn-width",
        type=float,
        default=24.0,
        help="Figure width in inches for STN plot (default: 24).",
    )
    parser.add_argument(
        "--stn-height",
        type=float,
        default=10.0,
        help="Figure height in inches for STN plot (default: 10).",
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

    deadlines: list[float] = []
    if is_economic_context(filepath, model_name):
        deadlines = read_deadlines(toml_path)
        for _i, _dh in enumerate(deadlines):
            print(f"  Deadline{_i + 1}: {_dh} h ({_dh / 24:.1f} d)")

    has_production = bool(data.get("production"))

    main_width = args.fig_width
    main_height = (
        args.fig_height
        if args.fig_height is not None
        else (8.0 if args.publish else 10.0)
    )

    if args.publish:
        fig_main = plt.figure(figsize=(main_width, main_height), layout="constrained")
        fig_main.set_constrained_layout_pads(
            w_pad=0.01,
            h_pad=0.01,
            wspace=0.02,
            hspace=0.02,
        )
        ax_gantt = fig_main.add_subplot(111)

        if has_production:
            fig_prod = plt.figure(
                figsize=(main_width, args.prod_height), layout="constrained"
            )
            fig_prod.set_constrained_layout_pads(
                w_pad=0.01,
                h_pad=0.01,
                wspace=0.02,
                hspace=0.02,
            )
            ax_prod = fig_prod.add_subplot(111)
        else:
            fig_prod = None
            ax_prod = None
    else:
        fig_main = plt.figure(figsize=(main_width, main_height), layout="constrained")
        fig_main.set_constrained_layout_pads(
            w_pad=0.01,
            h_pad=0.01,
            wspace=0.02,
            hspace=0.02,
        )
        if has_production:
            gs = fig_main.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
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
        data,
        ax_gantt,
        deadlines=deadlines,
        publish_mode=args.publish,
        no_title=args.no_title,
        no_break=args.no_break,
        max_breaks=args.max_breaks,
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

    default_ext = (
        "svg" if args.publish and args.svg else "pdf" if args.publish else "png"
    )
    save_main_requested = (
        args.no_show
        or args.output is not None
        or args.publish
        or args.production_output is not None
    )

    if save_main_requested:
        if args.publish:
            gantt_out = build_output_path(
                f"{filepath.stem}_gantt",
                args.output,
                default_ext,
                args.publish,
                publish_ext=default_ext,
            )
            fig_main.savefig(gantt_out, dpi=800, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved Gantt chart to {gantt_out}")

            if fig_prod is not None:
                prod_out = build_output_path(
                    f"{filepath.stem}_production",
                    args.production_output,
                    default_ext,
                    args.publish,
                    publish_ext=default_ext,
                )
                fig_prod.savefig(
                    prod_out, dpi=800, bbox_inches="tight", pad_inches=0.02
                )
                print(f"Saved production chart to {prod_out}")
        else:
            prod_out = build_output_path(
                f"{filepath.stem}_gantt",
                args.output,
                default_ext,
                args.publish,
                publish_ext=default_ext,
            )
            fig_main.savefig(prod_out, dpi=800, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved combined chart to {prod_out}")

    if args.stn:
        if not args.toml:
            print(f"Auto-selected STN TOML: {toml_path}")
        if not toml_path.exists():
            print(f"Error: TOML file '{toml_path}' not found. STN graph skipped.")
        else:
            fig_stn = plt.figure(
                figsize=(args.stn_width, args.stn_height), layout="constrained"
            )
            ax_stn = fig_stn.add_subplot(111)
            plot_stn_graph(str(toml_path), ax_stn, show_dsch=not args.hide_dsch)
            if args.no_show or args.stn_output or args.publish:
                stn_out = build_output_path(
                    f"{toml_path.stem}_stn",
                    args.stn_output,
                    default_ext,
                    args.publish,
                    publish_ext=default_ext,
                )
                fig_stn.savefig(stn_out, dpi=450, bbox_inches="tight", pad_inches=0.02)
                print(f"Saved STN graph to {stn_out}")

    if not (
        save_main_requested
        or args.stn_output is not None
        or (args.publish and args.stn)
    ):
        plt.show()


if __name__ == "__main__":
    main()
