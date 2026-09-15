# type: ignore
"""
Wine Scheduling Economic Optimization Model using Pyomo GDP.

This script keeps the GDP scheduling formulation and switches the objective
to economic profit maximization using price, outsourcing, and lateness terms.
"""

import math
from itertools import combinations
import pyomo.environ as pyo
import pyomo.gdp as gdp
import tomllib
from pyomo.opt import SolverFactory

from utils import (
    export_results,
    restore_variable_values,
    snapshot_variable_values,
)

transformation_type = "bigm"
solver_name = "gurobi_persistent"


def add_cover_cuts(model):
    """
    Cover-cut tightening of the LP relaxation.

    For each (i, n) adds:
        b[i, n] <= sum_j( Bmax[i, j] * y[i, j, n] )

    Follows directly from A04  (b[i,n] = sum_j bj[i,j,n])  and the
    disjunctive bounds  bj[i,j,n] <= Bmax[i,j]  when y[i,j,n]=1  and
    bj[i,j,n] = 0  when y[i,j,n]=0.  No feasible integer point is cut.

    After the bigm transformation the disjunctive bounds become
        bj <= Bmax + M*(1 - y)

    where M may be loose.  The cover cut gives the LP a direct, tight
    linear relationship between b and the y binaries without depending
    on the solver to derive it from the bigm expansion.
    """
    compatible_by_task: dict = {}
    cover_indices = []

    for i in model.i:
        # Pool tasks have no y variables, cover cut does not apply.
        pairs = [
            (j, round(pyo.value(model.Bmax[i, j]), 6))
            for j in model.j
            if (i, j) in model.ij_nonpool
        ]
        if not pairs:
            continue
        compatible_by_task[i] = pairs
        for n in model.n:
            cover_indices.append((i, n))

    if not cover_indices:
        print(
            "  [cover_cut] No compatible (task, unit) pairs found; "
            "no cover-cut constraints added."
        )
        return

    def cover_rule(m, i, n):
        return m.b[i, n] <= sum(
            bmax_val * m.y[i, j, n] for j, bmax_val in compatible_by_task[i]
        )

    model.cover_cut = pyo.Constraint(cover_indices, rule=cover_rule)
    print(
        f"  [cover_cut] Added {len(cover_indices)} cover-cut constraints "
        f"across {len(compatible_by_task)} tasks."
    )


DEFAULT_PERFORMANCE_OPTIONS = {
    # The integral fresh/reuse flow is already an exact physical-vessel path
    # formulation for barriques. The generic cumulative core is therefore
    # redundant for integer feasibility in that pool.
    "barrique_flow_only": True,
    # A pair relation has four states and can be represented exactly by two
    # binary code bits instead of four one-hot binaries (Big-M mode only).
    "compact_jar_relations": False,
    # Use each task's proven vessel-count upper bound in every linking row.
    "tight_pool_bounds": True,
    # Event numbers are task-local counters, so a capacity sum over a common
    # event number is not a valid time-capacity inequality.
    "remove_event_index_capacity": True,
    # Exact comparator formulations. These disaggregate the corresponding
    # integer pool counts into explicitly named physical vessels while leaving
    # the task/event, material-balance, timing, and objective model unchanged.
    "individual_barriques": False,
    "individual_jars": False,
}


def create_wine_scheduling_model(
    toml_file="parameters.toml",
    performance_options=None,
):
    """Create and reformulate the economic GDP model."""

    perf = dict(DEFAULT_PERFORMANCE_OPTIONS)
    if performance_options:
        unknown = set(performance_options) - set(perf)
        if unknown:
            raise ValueError(f"Unknown performance option(s): {sorted(unknown)}")
        perf.update(performance_options)

    def task_stage(task_name):
        return task_name.rstrip("0123456789")

    def task_line(task_name):
        i = len(task_name)
        while i > 0 and task_name[i - 1].isdigit():
            i -= 1
        return task_name[i:]

    def is_aging_stage(stage_name):
        return stage_name.startswith("Age")

    def unit_order_key(unit_name):
        if "#" in unit_name:
            _, suffix = unit_name.rsplit("#", 1)
            try:
                return int(suffix)
            except ValueError:
                return 10**9
        return 1

    # Load parameters from TOML file
    with open(toml_file, "rb") as f:
        params = tomllib.load(f)

    model = pyo.ConcreteModel(name="WineSchedulingEconomicGDP")
    model._performance_options = dict(perf)

    # ========================================
    # SETS
    # ========================================

    # Generate tasks from per-line step definitions
    lines_cfg = params["lines"]
    lines = list(lines_cfg.keys())
    line_product = {}
    line_last_task = {}
    line_lateness_task = {}
    aging_task_by_product = {}
    tasks = []
    for line, cfg in lines_cfg.items():
        if "product" not in cfg:
            raise ValueError(f"Line '{line}' missing 'product' in [lines] section")
        if cfg["product"] not in params["products"]:
            raise ValueError(
                f"Line '{line}' references unknown product '{cfg['product']}'"
            )
        line_product[line] = cfg["product"]
        for step in cfg["steps"]:
            if step == "Pr":
                if "Pr" not in tasks:
                    tasks.append("Pr")
            else:
                tasks.append(f"{step}{line}")
        line_last_task[line] = f"{cfg['steps'][-1]}{line}"
        # Lateness is evaluated on the last mandatory processing step
        # Optional trailing Stg should not hide lateness when it is skipped
        if len(cfg["steps"]) >= 2 and cfg["steps"][-1] == "Stg":
            line_lateness_task[line] = f"{cfg['steps'][-2]}{line}"
        else:
            line_lateness_task[line] = line_last_task[line]
        # Record the aging task regardless of its position in the step list
        for step in cfg["steps"]:
            if is_aging_stage(step):
                aging_task_by_product[cfg["product"]] = f"{step}{line}"
                break

    # Derived line groups (used for states, ICS/IPS and tc1)
    no_fl_lines = {ln for ln, cfg in lines_cfg.items() if "Fl" not in cfg["steps"]}
    # stg_int_lines: lines where Stg is an intermediate buffer before an aging step
    stg_int_lines = {
        ln
        for ln, cfg in lines_cfg.items()
        if "Stg" in cfg["steps"] and cfg["steps"][-1] != "Stg"
    }
    eco_lines = {ln for ln, cfg in lines_cfg.items() if cfg.get("ecobulk", False)}
    # Lines where an aging step (AgeBar*, AgeJar*) precedes Cs
    aging_before_cs_lines = {
        ln
        for ln, cfg in lines_cfg.items()
        if "Cs" in cfg["steps"]
        and any(
            is_aging_stage(step) for step in cfg["steps"][: cfg["steps"].index("Cs")]
        )
    }

    # Two-path aging for stg_int_lines: direct ({AgeStep}{line}) already in tasks,
    # and Stg-variant ({AgeStep}Stg{line}) that consumes vbuf instead of vl/v.
    # stg_variant_base maps Stg-variant task name -> base aging stage for template lookup.
    stg_variant_base: dict[str, str] = {}
    for ln in stg_int_lines:
        for step in lines_cfg[ln]["steps"]:
            if is_aging_stage(step):
                stg_task = f"{step}Stg{ln}"
                tasks.append(stg_task)
                stg_variant_base[stg_task] = step
                break

    def get_stage(task_name):
        """Return the template stage for a task, resolving Stg-variant aliases."""
        return stg_variant_base.get(task_name, task_stage(task_name))

    # aging_tasks_by_product: product -> list of all aging tasks (direct + Stg-variant)
    aging_tasks_by_product: dict[str, list[str]] = {
        prod: [task] for prod, task in aging_task_by_product.items()
    }
    for stg_t, base_stage in stg_variant_base.items():
        prod = line_product[task_line(stg_t)]
        aging_tasks_by_product[prod].append(stg_t)

    # A line with intermediate storage has two routes:
    # - A direct route that bypasses Stg,
    # - A buffered route that uses the Stg-specific aging task.
    # Keeping the alternatives explicit is important for deriving
    # valid task/event and time-window bounds below.
    line_task_paths: dict[str, list[list[str]]] = {}
    for ln, cfg in lines_cfg.items():
        if ln not in stg_int_lines:
            line_task_paths[ln] = [
                ["Pr" if step == "Pr" else f"{step}{ln}" for step in cfg["steps"]]
            ]
            continue
        aging_step = next(step for step in cfg["steps"] if is_aging_stage(step))
        direct_path = [
            "Pr" if step == "Pr" else f"{step}{ln}"
            for step in cfg["steps"]
            if step != "Stg"
        ]
        buffered_path = [
            "Pr"
            if step == "Pr"
            else (f"{aging_step}Stg{ln}" if step == aging_step else f"{step}{ln}")
            for step in cfg["steps"]
        ]
        line_task_paths[ln] = [direct_path, buffered_path]

    model.i = pyo.Set(initialize=tasks)
    model.iStgInt = pyo.Set(initialize=[f"Stg{ln}" for ln in stg_int_lines])
    model.inst = pyo.Set(initialize=tasks)
    model.ipst = pyo.Set(initialize=list(line_lateness_task.values()))
    model.inpst = pyo.Set(
        initialize=[t for t in tasks if t.startswith(("Pr", "Fa", "Fl"))]
    )

    units_cfg = params.get("units", {})
    if not units_cfg:
        raise ValueError("Missing required section [units] in TOML")

    unit_instances = []
    unit_instances_by_base = {}
    storage_units = []
    exterior_units = []
    ecobulk_units = []

    for unit_name, unit_cfg in units_cfg.items():
        quantity = int(unit_cfg.get("quantity", 1))
        if quantity < 1:
            raise ValueError(f"Unit '{unit_name}' must define quantity >= 1")
        unit_instances_by_base[unit_name] = []
        for idx in range(1, quantity + 1):
            inst_name = unit_name if quantity == 1 else f"{unit_name}#{idx}"
            unit_instances.append(inst_name)
            unit_instances_by_base[unit_name].append(inst_name)
            if "Stg" in unit_cfg.get("task_bounds", {}):
                storage_units.append(inst_name)
            if unit_cfg.get("exterior", False):
                exterior_units.append(inst_name)
            if unit_cfg.get("ecobulk", False):
                ecobulk_units.append(inst_name)

    model.j = pyo.Set(initialize=unit_instances)
    model.JST = pyo.Set(initialize=storage_units)
    model.JEXT = pyo.Set(initialize=exterior_units)
    model.JECO = pyo.Set(initialize=ecobulk_units)
    model.JBAR = pyo.Set(initialize=unit_instances_by_base.get("barrique", []))
    model.JJAR = pyo.Set(initialize=unit_instances_by_base.get("jar", []))

    # Task-unit pairs (ij)
    ij_data = []
    for task in tasks:
        prefix = get_stage(task)

        if (
            prefix in params["templates"]
            and "compatible_units" in params["templates"][prefix]
        ):
            for unit in params["templates"][prefix]["compatible_units"]:
                if unit not in units_cfg:
                    raise ValueError(
                        f"Template '{prefix}' references unknown unit '{unit}'"
                    )
                if prefix not in units_cfg[unit].get("task_bounds", {}):
                    raise ValueError(
                        f"Unit '{unit}' has no task_bounds entry for task '{prefix}'"
                    )
                for unit_inst in unit_instances_by_base[unit]:
                    ij_data.append((task, unit_inst))
    model.ij = pyo.Set(initialize=ij_data, dimen=2)

    barr_set = set(unit_instances_by_base.get("barrique", []))
    jar_set = set(unit_instances_by_base.get("jar", []))

    def compat(task):
        return [j for i, j in ij_data if i == task]

    barr_tasks = [
        i for i in tasks if compat(i) and all(j in barr_set for j in compat(i))
    ]
    jar_tasks = [i for i in tasks if compat(i) and all(j in jar_set for j in compat(i))]
    pool_tasks = set(barr_tasks + jar_tasks)

    model.iBAR = pyo.Set(initialize=barr_tasks)
    model.iJAR = pyo.Set(initialize=jar_tasks)

    ij_nonpool_data = [(i, j) for (i, j) in ij_data if i not in pool_tasks]
    model.ij_nonpool = pyo.Set(initialize=ij_nonpool_data, dimen=2)

    def fixed_pool_capacity(unit_name):
        unit_cfg = units_cfg.get(unit_name)
        if unit_cfg is None:
            raise ValueError(f"Missing required pool unit '{unit_name}' in [units]")
        task_bounds = unit_cfg.get("task_bounds", {})
        if not task_bounds:
            raise ValueError(
                f"Pool unit '{unit_name}' must define at least one task_bounds entry"
            )

        capacities = set()
        for task_name, limits in task_bounds.items():
            if isinstance(limits, (int, float)):
                bmin_val = bmax_val = float(limits)
            else:
                bmin_val, bmax_val = float(limits[0]), float(limits[1])
            if abs(bmax_val - bmin_val) > 1e-6:
                raise ValueError(
                    f"Pool unit '{unit_name}' requires fixed bounds for '{task_name}', "
                    f"got [{bmin_val}, {bmax_val}]"
                )
            capacities.add(round(bmax_val, 6))

        if len(capacities) != 1:
            raise ValueError(
                f"Pool unit '{unit_name}' must use one fixed capacity across tasks, "
                f"found {sorted(capacities)}"
            )
        return capacities.pop()

    barr_cap = fixed_pool_capacity("barrique") if barr_set else 0.0
    jar_cap = fixed_pool_capacity("jar") if jar_set else 0.0
    n_barr = len(barr_set)
    n_jar = len(jar_set)
    model._n_barr = n_barr
    model._n_jar = n_jar

    model.n = pyo.Set(initialize=list(range(1, params["global"]["n_max"] + 1)))

    # States:
    # - s_red / s_white_rose: shared raw-material pools by product category
    # - m: shared liquid must pool
    # - v{l}: after Fa (zero-wait intermediate)
    # - vl{l}: after Fl (NIS intermediate, only when Fl is in steps)
    # - product state: final product key declared on each line
    # - dsch: discard balance state
    line_raw_state = {}
    raw_states = set()
    has_pressing = any("Pr" in cfg["steps"] for cfg in lines_cfg.values())
    states = ["dsch"] + (["grape_skin"] if has_pressing else [])
    for line, cfg in lines_cfg.items():
        product_key = line_product[line]
        category = params["products"][product_key].get("category", "").strip().lower()
        raw_state = "s_red" if category == "red" else "s_white_rose"
        line_raw_state[line] = raw_state
        raw_states.add(raw_state)
        if raw_state not in states:
            states.append(raw_state)
        if "Pr" in cfg["steps"] and "m" not in states:
            states.append("m")
        states.append(f"v{line}")
        if "Fl" in cfg["steps"]:
            states.append(f"vl{line}")
        if line in aging_before_cs_lines:
            states.append(f"va{line}")
        if line in stg_int_lines:
            states.append(f"vbuf{line}")
        states.append(line_product[line])
    model.s = pyo.Set(initialize=states)

    # States logic
    IPS_data = []
    ICS_data = []
    for task in tasks:
        line = task_line(task)
        stage = task_stage(task)
        if stage == "Pr":
            # Shared Pr task: all pressing lines use s_white_rose
            ICS_data.append((task, "s_white_rose"))
            IPS_data.append((task, "m"))
            IPS_data.append((task, "dsch"))
            IPS_data.append((task, "grape_skin"))
        elif stage == "Fa":
            if "Pr" in lines_cfg[line]["steps"]:
                ICS_data.append((task, "m"))
            else:
                ICS_data.append((task, line_raw_state[line]))
            IPS_data.append((task, f"v{line}"))
            IPS_data.append((task, "dsch"))
        elif stage == "Fl":
            ICS_data.append((task, f"v{line}"))
            IPS_data.append((task, f"vl{line}"))
            IPS_data.append((task, "dsch"))
        elif stage == "Cs":
            if line in aging_before_cs_lines:
                # Aging precedes Cs, consume the post-aging intermediate va{line}
                ICS_data.append((task, f"va{line}"))
            elif line in no_fl_lines:
                ICS_data.append((task, f"v{line}"))
            else:
                ICS_data.append((task, f"vl{line}"))
            IPS_data.append((task, line_product[line]))
        elif stage == "Stg":
            if line in stg_int_lines:
                # Intermediate buffer: consume pre-aging state, produce vbuf (can wait)
                pre_ag = f"v{line}" if line in no_fl_lines else f"vl{line}"
                ICS_data.append((task, pre_ag))
                IPS_data.append((task, f"vbuf{line}"))
        elif is_aging_stage(stage):
            if line in aging_before_cs_lines:
                # Two-path aging for stg_int_lines:
                #   Stg-variant task -> consume vbuf (held by Stg intermediate step)
                #   Direct aging task -> consume pre_ag (vl or v, same as non-stg lines)
                if task in stg_variant_base:
                    ICS_data.append((task, f"vbuf{line}"))
                elif line in stg_int_lines:
                    pre_ag = f"v{line}" if line in no_fl_lines else f"vl{line}"
                    ICS_data.append((task, pre_ag))
                elif line in no_fl_lines:
                    ICS_data.append((task, f"v{line}"))
                else:
                    ICS_data.append((task, f"vl{line}"))
                IPS_data.append((task, f"va{line}"))
            else:
                ICS_data.append((task, line_product[line]))
                IPS_data.append((task, line_product[line]))
    model.IPS = pyo.Set(initialize=IPS_data, dimen=2)
    model.ICS = pyo.Set(initialize=ICS_data, dimen=2)

    model.SR = pyo.Set(initialize=sorted(raw_states))
    model.line_raw_state = line_raw_state
    model.SP = pyo.Set(
        initialize=["dsch"]
        + (["grape_skin"] if has_pressing else [])
        + [line_product[ln] for ln in lines]
    )
    model.SI = model.s - model.SP - model.SR
    model.SFISEco = pyo.Set(initialize=[line_product[ln] for ln in eco_lines])
    model.SFISBar = pyo.Set(
        initialize=[line_product[ln] for ln in lines if ln not in eco_lines]
    )
    model.SZW = pyo.Set(
        initialize=[f"v{ln}" for ln in lines]
        + (["m"] if any("Pr" in lines_cfg[ln]["steps"] for ln in lines) else [])
        + [f"vbuf{ln}" for ln in stg_int_lines]
    )
    model.SNIS = pyo.Set(
        initialize=[f"vl{ln}" for ln in lines if ln not in no_fl_lines]
        + [f"va{ln}" for ln in aging_before_cs_lines]
    )
    model.SD = pyo.Set(
        initialize=sorted(set(model.SI) - set(model.SZW) - set(model.SNIS))
    )

    tc1_data = []
    for line in lines:
        steps = lines_cfg[line]["steps"]
        for idx in range(len(steps) - 1):
            current_task = f"{steps[idx]}{line}"
            next_task = f"{steps[idx + 1]}{line}"
            if steps[idx + 1] == "Stg" and line in stg_int_lines:
                pass
            elif steps[idx] == "Stg" and line in stg_int_lines:
                pass
            elif is_aging_stage(steps[idx]) and line in stg_int_lines:
                pass
            elif steps[idx] == "Pr" and steps[idx + 1] == "Fa":
                pass
            else:
                tc1_data.append((current_task, next_task))
    # Stg -> AgeViaStg: if Stg runs at event n, AgeViaStg must run at event n+1.
    # vbuf is ZW so timing is tight, and tc1 enforces the binary coupling.
    for stg_t in stg_variant_base:
        ln = task_line(stg_t)
        tc1_data.append((f"Stg{ln}", stg_t))
    model.tc1 = pyo.Set(initialize=tc1_data, dimen=2)
    model.prd = pyo.Set(initialize=[line_product[ln] for ln in lines])

    # ========================================
    # PARAMETERS
    # ========================================
    global_cfg = params.get("global", {})
    model.iMax = pyo.Param(initialize=global_cfg.get("iMax", 1))
    model.jMax = pyo.Param(initialize=global_cfg.get("jMax", 1))
    model.jMin = pyo.Param(initialize=global_cfg.get("jMin", 1))
    model.ST0 = pyo.Param(model.s, initialize=params["initial_inventory"], default=0)
    model.STmax = pyo.Param(model.s, initialize=0)
    model.alpha = pyo.Param(model.i, mutable=True, default=0)
    model.beta = pyo.Param(model.i, mutable=True, default=0)
    model.Bmin = pyo.Param(model.ij, mutable=True, default=0)
    model.Bmax = pyo.Param(model.ij, mutable=True, default=0)

    for task in model.i:
        prefix = get_stage(task)

        if prefix in params["templates"]:
            template = params["templates"][prefix]
            if "alpha" in template:
                model.alpha[task] = template["alpha"]
            model.beta[task] = template["beta"]
            if "compatible_units" in template:
                for unit in template["compatible_units"]:
                    limits = units_cfg[unit]["task_bounds"][prefix]
                    if isinstance(limits, (int, float)):
                        bmin_val = bmax_val = float(limits)
                    else:
                        bmin_val, bmax_val = float(limits[0]), float(limits[1])
                    for unit_inst in unit_instances_by_base[unit]:
                        if (task, unit_inst) in model.ij:
                            model.Bmin[task, unit_inst] = bmin_val
                            model.Bmax[task, unit_inst] = bmax_val

    # Cooling cost assumes beta=0 for all exterior-unit tasks (otherwise
    # it cannot be handled as MILP).
    bad_cooling_tasks = [
        i
        for i in model.inst
        if pyo.value(model.beta[i]) != 0
        and any((i, j) in model.ij and j in model.JEXT for j in model.j)
    ]
    if bad_cooling_tasks:
        raise ValueError(
            f"Cooling cost requires beta=0 for exterior-unit tasks, "
            f"but beta != 0 for: {bad_cooling_tasks}. "
        )

    model.H = pyo.Param(initialize=params["global"]["H"])

    model.rhoIScons = pyo.Param(model.i, model.s, mutable=True, default=0)
    model.rhoISprod = pyo.Param(model.i, model.s, mutable=True, default=0)

    if "rho" in params:
        for task_key, rho_data in params["rho"].items():
            if task_key in model.i:
                if "rhoIScons" in rho_data:
                    for state, value in rho_data["rhoIScons"].items():
                        model.rhoIScons[task_key, state] = value
                if "rhoISprod" in rho_data:
                    for state, value in rho_data["rhoISprod"].items():
                        model.rhoISprod[task_key, state] = value

    # Intermediate Stg rho: pass-through buffer (consume pre-aging, produce vbuf)
    for line in stg_int_lines:
        task = f"Stg{line}"
        pre_ag = f"v{line}" if line in no_fl_lines else f"vl{line}"
        model.rhoIScons[task, pre_ag] = -1.0
        model.rhoISprod[task, f"vbuf{line}"] = 1.0

    # Stg-variant aging rho: consume vbuf (not pre_ag), same yield as direct aging task.
    for stg_t, base_stage in stg_variant_base.items():
        ln = task_line(stg_t)
        direct_task = f"{base_stage}{ln}"
        pre_ag = f"v{ln}" if ln in no_fl_lines else f"vl{ln}"
        model.rhoIScons[stg_t, pre_ag] = 0.0  # no pre_ag consumption
        model.rhoIScons[stg_t, f"vbuf{ln}"] = -1.0  # consume vbuf instead
        va_state = f"va{ln}"
        model.rhoISprod[stg_t, va_state] = pyo.value(
            model.rhoISprod[direct_task, va_state]
        )

    imax_young_pre = int(global_cfg.get("iMaxYoungWine", 1))
    aging_prods_pre = set(aging_task_by_product.keys())
    demand_data = {
        k: v["demand"] * (imax_young_pre if k not in aging_prods_pre else 1)
        for k, v in params["products"].items()
    }
    model.D = pyo.Param(model.s, initialize=demand_data, default=0)
    model.line_product = line_product
    model.line_last_task = line_last_task
    model.line_lateness_task = line_lateness_task
    model.aging_task_by_product = aging_task_by_product
    model.product_meta = params["products"]
    model.SMarket = pyo.Set(initialize=[p for p in model.SP if p in params["products"]])

    price_data = {k: v.get("price", 0.0) for k, v in params["products"].items()}
    outsourcing_data = {
        k: v.get("costOutsourcing", 0.0) for k, v in params["products"].items()
    }
    outsourcing_allowed_data = {
        k: 1 if v.get("outsourced", True) else 0 for k, v in params["products"].items()
    }
    model.Price = pyo.Param(model.s, initialize=price_data, default=0.0)
    model.CostOutsourcing = pyo.Param(model.s, initialize=outsourcing_data, default=0.0)
    model.OutsourceAllowed = pyo.Param(
        model.s, initialize=outsourcing_allowed_data, default=1
    )
    model.penaltyLate = pyo.Param(initialize=global_cfg.get("penaltyLate", 0.0))

    aging_hours_by_product = {p: 0.0 for p in model.SMarket}
    for line in lines:
        product_key = line_product[line]
        for stage in lines_cfg[line]["steps"]:
            if is_aging_stage(stage):
                aging_hours_by_product[product_key] = float(
                    params["templates"][stage].get("alpha", 0.0)
                )
                break
    model.AgingHoursByProduct = pyo.Param(
        model.SMarket,
        initialize=aging_hours_by_product,
        default=0.0,
    )

    raw_material_cost_cfg = params.get("raw_material_cost", {})
    model.RawMaterialCostPerL = pyo.Param(
        model.SR,
        initialize=raw_material_cost_cfg,
        default=0.0,
    )

    model.penaltyMS = pyo.Param(initialize=params["global"].get("penaltyMS", 0.0))
    model.costCooling = pyo.Param(initialize=params["global"].get("costCooling", 0.0))
    model.penaltyDiscard = pyo.Param(initialize=global_cfg.get("penaltyDiscard", 3.0))
    model.stgHoldingCost = pyo.Param(initialize=global_cfg.get("stgHoldingCost", 0.05))
    model.GrapeSkinValue = pyo.Param(initialize=global_cfg.get("grape_skin_value", 0.0))
    imax_young = int(global_cfg.get("iMaxYoungWine", 1))

    # Deadline list
    if "deadlines" not in global_cfg:
        raise ValueError("Missing required [global].deadlines list in TOML")
    deadlines_val = [float(d) for d in global_cfg["deadlines"]]
    if len(deadlines_val) < imax_young:
        raise ValueError(
            f"[global].deadlines must contain at least {imax_young} values, "
            f"got {len(deadlines_val)}"
        )
    model.DeadlineK = pyo.Param(
        range(1, imax_young + 1),
        initialize={k + 1: deadlines_val[k] for k in range(imax_young)},
    )

    # Young wines
    aging_products_early = set(aging_task_by_product.keys())
    young_lines = [ln for ln in lines if line_product[ln] not in aging_products_early]
    dual_late_lines_set = set(young_lines) if imax_young > 1 else set()
    dual_late_products = sorted({line_product[ln] for ln in dual_late_lines_set})
    dual_late_products_set = set(dual_late_products)
    young_lines_by_product = {
        product: [ln for ln in young_lines if line_product[ln] == product]
        for product in dual_late_products
    }

    # Sparse Index Sets
    n_max = max(model.n)

    # Build precedence pairs from state-indexed producer/consumer maps
    producer_tasks_by_state = {s: [] for s in model.s}
    consumer_tasks_by_state = {s: [] for s in model.s}

    for i_prod, s in model.IPS:
        producer_tasks_by_state[s].append(i_prod)
    for i_cons, s in model.ICS:
        consumer_tasks_by_state[s].append(i_cons)

    precedence_pairs = [
        (i_cons, i_prod, s)
        for s in model.s
        for i_cons in consumer_tasks_by_state[s]
        for i_prod in producer_tasks_by_state[s]
        if i_cons != i_prod
    ]
    model.PrecedencePairs = pyo.Set(initialize=precedence_pairs, dimen=3)
    model.ZWPrecedencePairs = pyo.Set(
        initialize=[(i, ip, s) for i, ip, s in precedence_pairs if s in model.SZW],
        dimen=3,
    )
    model.NISPrecedencePairs = pyo.Set(
        initialize=[(i, ip, s) for i, ip, s in precedence_pairs if s in model.SNIS],
        dimen=3,
    )
    model.EcoPrecedencePairs = pyo.Set(
        initialize=[(i, ip, s) for i, ip, s in precedence_pairs if s in model.SFISEco],
        dimen=3,
    )

    # ========================================
    # VARIABLES
    # ========================================

    # y and bj only for non-pool tasks, since pool tasks use NumBarr/NumJar integer counts
    model.y = pyo.Var(model.ij_nonpool, model.n, domain=pyo.Binary)

    # Continuous Variables
    model.b = pyo.Var(model.i, model.n, domain=pyo.NonNegativeReals)
    model.bj = pyo.Var(model.ij_nonpool, model.n, domain=pyo.NonNegativeReals)

    # Set upper bounds for bj and b to allow BigM estimation
    # Find global max capacity
    max_tank_cap = 0
    if hasattr(model, "Bmax"):
        for key in model.Bmax:
            val = pyo.value(model.Bmax[key])
            if val > max_tank_cap:
                max_tank_cap = val
    if max_tank_cap == 0:
        max_tank_cap = 500000  # Fallback

    # Precompute: for tasks that directly produce a final market product,
    # map i -> (s, rho) so the product-ub-based b[i,n] UB can be applied below.
    product_ub_cfg = params.get("product_ub", {})
    market_prod_rho: dict[str, tuple[str, float]] = {}
    smarket_set = set(model.SMarket)
    for ip, sp in model.IPS:
        if sp in smarket_set and sp in product_ub_cfg:
            rho_val = pyo.value(model.rhoISprod[ip, sp])
            if rho_val > 0:
                market_prod_rho[ip] = (sp, rho_val)

    # Per-task batch upper bound, reused below for the global on/off link
    # b[i, n] <= b_ub[i] * W[i, n].
    b_ub: dict[str, float] = {}
    for i in model.i:
        if i in barr_tasks:
            b_ub_i = barr_cap * n_barr
        elif i in jar_tasks:
            b_ub_i = jar_cap * n_jar
        else:
            stage_i = get_stage(i)
            template_i = params.get("templates", {}).get(stage_i, {})
            jmax_i = int(template_i.get("max_units", pyo.value(model.jMax)))
            bmax_vals_i = sorted(
                [pyo.value(model.Bmax[i, j]) for j in model.j if (i, j) in model.ij],
                reverse=True,
            )
            b_ub_i = sum(bmax_vals_i[:jmax_i]) if bmax_vals_i else max_tank_cap
            if b_ub_i <= 0:
                b_ub_i = max_tank_cap
        if i in market_prod_rho:
            sp, rho_val = market_prod_rho[i]
            b_ub_i = min(b_ub_i, product_ub_cfg[sp] / rho_val)
        b_ub[i] = b_ub_i

    # Propagate batch bounds through proven one-to-one zero-inventory arcs on
    # young-wine lines. If producer ``ip`` and consumer ``ic`` are the unique
    # users of state ``s``, h14/h15 and A13a imply, event by event,
    #
    #  -rho_cons[ic,s] * b[ic,n+1] = rho_prod[ip,s] * b[ip,n] - Discard[s,n+1].
    #
    # Since discard is nonnegative, an upstream bound can safely tighten the
    # downstream batch.
    tc1_set = set(tc1_data)
    zero_inventory_states = set(model.SZW) | set(model.SNIS)
    young_line_tasks = {
        ln: ["Pr" if step == "Pr" else f"{step}{ln}" for step in lines_cfg[ln]["steps"]]
        for ln in young_lines
    }
    young_zero_wait_arcs: list[tuple[str, str, str, float, float]] = []
    for path in young_line_tasks.values():
        for ip, ic in zip(path, path[1:]):
            if (ip, ic) not in tc1_set:
                continue
            shared_states = [
                s
                for s in zero_inventory_states
                if (ip, s) in model.IPS and (ic, s) in model.ICS
            ]
            if len(shared_states) != 1:
                continue
            s = shared_states[0]
            if producer_tasks_by_state[s] != [ip] or consumer_tasks_by_state[s] != [ic]:
                continue
            if abs(float(pyo.value(model.ST0[s]))) > 1e-9:
                continue
            rho_prod = float(pyo.value(model.rhoISprod[ip, s]))
            rho_cons = -float(pyo.value(model.rhoIScons[ic, s]))
            if rho_prod > 0 and rho_cons > 0:
                young_zero_wait_arcs.append((ip, ic, s, rho_prod, rho_cons))

    changed = True
    while changed:
        changed = False
        for ip, ic, _s, rho_prod, rho_cons in young_zero_wait_arcs:
            consumer_ub = rho_prod * b_ub[ip] / rho_cons
            if consumer_ub < b_ub[ic] - 1e-9:
                b_ub[ic] = consumer_ub
                changed = True

    model._batch_ub = dict(b_ub)
    model._young_zero_wait_arcs = tuple(young_zero_wait_arcs)
    for i in model.i:
        for n in model.n:
            model.b[i, n].setub(b_ub[i])

    for i, j in model.ij_nonpool:
        for n in model.n:
            model.bj[i, j, n].setub(pyo.value(model.Bmax[i, j]))

    model.ST = pyo.Var(model.s, model.n, domain=pyo.NonNegativeReals)
    model.Ts = pyo.Var(
        model.i, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    model.Tf = pyo.Var(
        model.i, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    j_nonpool = [j for j in unit_instances if j not in barr_set and j not in jar_set]
    model.Tsj = pyo.Var(
        j_nonpool, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    model.Tfj = pyo.Var(
        j_nonpool, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )

    model.NumBarr = pyo.Var(
        model.iBAR, model.n, domain=pyo.NonNegativeIntegers, bounds=(0, n_barr)
    )
    model.NumJar = pyo.Var(
        model.iJAR, model.n, domain=pyo.NonNegativeIntegers, bounds=(0, n_jar)
    )

    # G1: tighten per-task pool-vessel upper bounds from the product cap.
    # A line cannot turn more vessels into sellable wine than product_ub allows:
    #   NumBarr[i,n] <= ceil(product_ub[s] / (rho_prod_i * barr_cap)).
    # Shrinks the domains from the full pool.
    def _pool_task_ub(task, cap, pool_size):
        prod = line_product.get(task_line(task))
        pub = product_ub_cfg.get(prod)
        if pub is None:
            return pool_size
        rho = max(
            (
                pyo.value(model.rhoISprod[task, s])
                for (ii, s) in model.IPS
                if ii == task and pyo.value(model.rhoISprod[task, s]) > 0
            ),
            default=1.0,
        )
        if rho <= 0:
            return pool_size
        return min(pool_size, max(1, math.ceil(pub / (rho * cap))))

    for task in model.iBAR:
        ub = _pool_task_ub(task, barr_cap, n_barr)
        for n in model.n:
            model.NumBarr[task, n].setub(ub)
    for task in model.iJAR:
        ub = _pool_task_ub(task, jar_cap, n_jar)
        for n in model.n:
            model.NumJar[task, n].setub(ub)

    model.FinalProd = pyo.Var(model.SP, domain=pyo.NonNegativeReals)
    model.Outsource = pyo.Var(model.SMarket, domain=pyo.NonNegativeReals)
    model.MS = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, model.H))
    all_late_products = sorted(
        {
            line_product[ln]
            for ln in lines
            if line_product[ln] not in aging_products_early
        }
    )
    late_camp_index = [
        (s, k)
        for k in range(1, imax_young + 1)
        for s in (all_late_products if k == 1 else dual_late_products)
    ]
    model.Late = pyo.Var(late_camp_index, domain=pyo.NonNegativeReals)
    campaign_products = list(dual_late_products)
    campaign_index = [
        (s, k) for s in campaign_products for k in range(1, imax_young + 1)
    ]
    model.CampaignProd = pyo.Var(campaign_index, domain=pyo.NonNegativeReals)
    model.CampaignOutsource = pyo.Var(campaign_index, domain=pyo.NonNegativeReals)
    for s, k in campaign_index:
        model.CampaignProd[s, k].setub(float(product_ub_cfg[s]))
        model.CampaignOutsource[s, k].setub(
            float(params["products"][s]["demand"])
            if pyo.value(model.OutsourceAllowed[s]) == 1
            else 0.0
        )
    model.Discard = pyo.Var(model.SI, model.n, domain=pyo.NonNegativeReals)
    nondiscardable_states = set()
    for s in model.SI:
        if s in nondiscardable_states:
            for n in model.n:
                model.Discard[s, n].fix(0.0)
            continue
        discard_ub = max(
            (
                pyo.value(model.rhoISprod[i, s])
                * sum(
                    pyo.value(model.Bmax[i, j]) for j in model.j if (i, j) in model.ij
                )
                for i in model.i
                if (i, s) in model.IPS
            ),
            default=0.0,
        )
        if discard_ub > 0:
            for n in model.n:
                model.Discard[s, n].setub(discard_ub)

    model._nondiscardable_states = frozenset(nondiscardable_states)
    model.task_jmax = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.jMax)
    model.task_jmin = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.jMin)
    model.task_imax = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.iMax)
    pressing_lines_pr = [ln for ln in lines if "Pr" in lines_cfg[ln]["steps"]]
    if "Pr" in model.i and pressing_lines_pr:
        n_young_pressing = sum(1 for ln in pressing_lines_pr if ln in set(young_lines))
        n_aged_pressing = len(pressing_lines_pr) - n_young_pressing
        min_K_pr = min(len(lines_cfg[ln]["steps"]) for ln in pressing_lines_pr)
        n_valid_pr = n_max - (min_K_pr - 1)
        iMax_val = int(pyo.value(model.iMax))
        model.task_imax["Pr"] = min(
            n_young_pressing * imax_young + n_aged_pressing * iMax_val, n_valid_pr
        )

    if imax_young > 1:
        young_task_names = [
            f"{step}{ln}"
            for ln in young_lines
            for step in lines_cfg[ln]["steps"]
            if step != "Pr"
        ]
        for t in young_task_names:
            if t in model.i:
                model.task_imax[t] = imax_young
        print(
            f"  [iMaxYoungWine={imax_young}] Applied to {len(young_task_names)} young-wine tasks (no aging)."
        )
    for task in model.i:
        stage = get_stage(task)
        template = params.get("templates", {}).get(stage, {})
        if "max_units" in template:
            model.task_jmax[task] = int(template["max_units"])
        if "min_units" in template:
            model.task_jmin[task] = int(template["min_units"])

    # The two aging routes are alternatives, not independent activation budgets.
    # Cs and the sum of direct/buffered aging activations share the line-level
    # iMax limit (AgingPathMax is added with the activation constraints below).
    for ln in stg_int_lines:
        cs_task = f"Cs{ln}"
        if cs_task not in model.i:
            continue
        model.task_imax[cs_task] = int(pyo.value(model.iMax))

    sr_set = set(model.SR)
    smarket_set_local = set(model.SMarket)
    for s in model.s:
        if s in sr_set:
            ub = float(pyo.value(model.ST0[s]))
        elif s in smarket_set_local:
            scale = imax_young if s not in aging_products_early else 1
            ub = float(product_ub_cfg.get(s, pyo.value(model.H))) * scale
        else:
            producers = [i for i in model.i if (i, s) in model.IPS]
            if producers:
                ub = sum(
                    max(
                        (
                            pyo.value(model.Bmax[i, j])
                            for j in model.j
                            if (i, j) in model.ij
                        ),
                        default=0.0,
                    )
                    * pyo.value(model.rhoISprod[i, s])
                    * float(pyo.value(model.task_imax[i]))
                    for i in producers
                )
                ub = max(ub, 0.0)
            else:
                ub = float(pyo.value(model.H))
        for n in model.n:
            model.ST[s, n].setub(ub)
    print(f"  [ST bounds] Applied upper bounds on ST for {len(list(model.s))} states.")

    # A makespan lower bound is valid only for lines that must produce in-house.
    required_inhouse_lines = [
        ln
        for ln in lines
        if pyo.value(model.D[line_product[ln]]) > 0
        and pyo.value(model.OutsourceAllowed[line_product[ln]]) == 0
    ]
    min_makespan = max(
        (
            min(
                sum(pyo.value(model.alpha[task]) for task in path)
                for path in line_task_paths[line]
            )
            for line in required_inhouse_lines
        ),
        default=0.0,
    )
    model.MS.setlb(min_makespan)

    FinalProd_bounds = params["product_ub"]
    for sp in model.SP:
        if sp in FinalProd_bounds:
            scale = imax_young if sp not in aging_products_early else 1
            model.FinalProd[sp].setub(FinalProd_bounds[sp] * scale)

    H_val = pyo.value(model.H)
    for s in model.SMarket:
        model.Outsource[s].setub(pyo.value(model.D[s]))
    for k in range(1, imax_young + 1):
        dk_val = deadlines_val[k - 1]
        prod_set = all_late_products if k == 1 else dual_late_products
        for s in prod_set:
            model.Late[s, k].setub(max(0.0, H_val - dk_val))
    # Per-task time windows over all legitimate line routes.
    # ES is the minimum prefix and LF the maximum suffix-derived finish bound
    # over every occurrence.
    H_val = pyo.value(model.H)
    task_occurrences: dict[str, list[tuple[float, float, int, int]]] = {
        task: [] for task in model.i
    }
    for paths in line_task_paths.values():
        for path in paths:
            alphas = [pyo.value(model.alpha[task]) for task in path]
            for pos, task in enumerate(path):
                prefix = sum(alphas[:pos])
                suffix = sum(alphas[pos + 1 :])
                task_occurrences[task].append(
                    (prefix, H_val - suffix, pos + 1, len(path))
                )

    task_ES = {
        task: min(occ[0] for occ in occurrences)
        for task, occurrences in task_occurrences.items()
    }
    task_LF = {
        task: max(occ[1] for occ in occurrences)
        for task, occurrences in task_occurrences.items()
    }
    for task in model.i:
        es = task_ES[task]
        lf = task_LF[task]
        for n in model.n:
            if es > 0:
                model.Ts[task, n].setlb(es)
            if lf < H_val:
                model.Tf[task, n].setub(lf)
                model.Ts[task, n].setub(lf)

    # Per-unit time-window bounds: Tsj/Tfj inherit the tightest [min ES, max LF]
    # across the compatible tasks of that unit.
    for j in j_nonpool:
        compat_tasks = [i for i in model.i if (i, j) in model.ij_nonpool]
        if not compat_tasks:
            continue
        lb_j = min(task_ES.get(t, 0.0) for t in compat_tasks)
        ub_j = max(task_LF.get(t, H_val) for t in compat_tasks)
        for n in model.n:
            if lb_j > 0:
                model.Tsj[j, n].setlb(lb_j)
                model.Tfj[j, n].setlb(lb_j)
            if ub_j < H_val:
                model.Tsj[j, n].setub(ub_j)
                model.Tfj[j, n].setub(ub_j)

    # ========================================
    # DISJUNCT PRE-CREATION
    # ========================================
    # W_{i,n} = active_disjuncts[i, n].binary_indicator_var
    # The GDP disjunct binary IS the task-activation variable. Pre-creating all
    # disjuncts here makes binary_indicator_var available for the algebraic
    # constraints below, no separate model.W Var or link_w constraints needed.
    active_disjuncts = {}
    inactive_disjuncts = {}
    for i in model.i:
        for n in model.n:
            d_act = gdp.Disjunct()
            d_inact = gdp.Disjunct()
            model.add_component(f"d_act_{i}_{n}", d_act)
            model.add_component(f"d_inact_{i}_{n}", d_inact)
            active_disjuncts[i, n] = d_act
            inactive_disjuncts[i, n] = d_inact

    # ========================================
    # REFORMULATION MODE
    # ========================================
    # The timing/precedence/pool/Ecobulk logic is emitted one of two ways from the
    # same disjunctions:
    # - "bigm" -> structure-aware Big-M written directly on the shared activation
    #             indicators (tight task-pair coefficients. No auxiliary variables),
    # - "hull" -> native Pyomo.GDP disjunctions, reformulated by gdp.hull.
    _hull = transformation_type == "hull"

    def guarded_relation(tag, bare_expr, guard_bins):
        """Emit `bare_expr` as a GDP disjunct active iff AND(guard_bins) holds.
        The conjunction of guard indicators is encoded with the exact
        AND-linearization on the disjunct's own indicator (hull path only)."""
        d_on = gdp.Disjunct()
        d_off = gdp.Disjunct()
        model.add_component(f"{tag}_on", d_on)
        model.add_component(f"{tag}_off", d_off)
        d_on.c = pyo.Constraint(expr=bare_expr)
        model.add_component(f"{tag}_dj", gdp.Disjunction(expr=[d_on, d_off]))
        Z = d_on.binary_indicator_var
        g = pyo.ConstraintList()
        model.add_component(f"{tag}_guard", g)
        for Y in guard_bins:
            g.add(Z <= Y)
        g.add(Z >= sum(guard_bins) - (len(guard_bins) - 1))

    def pool_seq(tag, i1, n1, i2, n2, rank):
        """Classify two pool task-events for the hull formulation.

        The two non-overlap branches allow an arbitrary idle gap. The overlap
        branches distinguish which task is already occupying vessels
        when the other starts. This is needed to enforce the
        cumulative pool capacity at every task start.
        """
        d_fwd = gdp.Disjunct()
        d_rev = gdp.Disjunct()
        d_ov_fwd = gdp.Disjunct()
        d_ov_rev = gdp.Disjunct()
        model.add_component(f"{tag}_fwd", d_fwd)
        model.add_component(f"{tag}_rev", d_rev)
        model.add_component(f"{tag}_ov_fwd", d_ov_fwd)
        model.add_component(f"{tag}_ov_rev", d_ov_rev)
        d_fwd.seq = pyo.Constraint(expr=model.Tf[i1, n1] <= model.Ts[i2, n2])
        d_fwd.rank_order = pyo.Constraint(expr=rank[i1, n1] + 1 <= rank[i2, n2])
        d_rev.seq = pyo.Constraint(expr=model.Tf[i2, n2] <= model.Ts[i1, n1])
        d_rev.rank_order = pyo.Constraint(expr=rank[i2, n2] + 1 <= rank[i1, n1])
        # i1 starts first and remains active when i2 starts.
        d_ov_fwd.start_order = pyo.Constraint(expr=model.Ts[i1, n1] <= model.Ts[i2, n2])
        d_ov_fwd.still_active = pyo.Constraint(
            expr=model.Ts[i2, n2] <= model.Tf[i1, n1]
        )
        d_ov_fwd.rank_order = pyo.Constraint(expr=rank[i1, n1] + 1 <= rank[i2, n2])
        # i2 starts first and remains active when i1 starts.
        d_ov_rev.start_order = pyo.Constraint(expr=model.Ts[i2, n2] <= model.Ts[i1, n1])
        d_ov_rev.still_active = pyo.Constraint(
            expr=model.Ts[i1, n1] <= model.Tf[i2, n2]
        )
        d_ov_rev.rank_order = pyo.Constraint(expr=rank[i2, n2] + 1 <= rank[i1, n1])
        model.add_component(
            f"{tag}_dj", gdp.Disjunction(expr=[d_fwd, d_rev, d_ov_fwd, d_ov_rev])
        )
        return d_ov_fwd.binary_indicator_var, d_ov_rev.binary_indicator_var

    # ========================================
    # STATIC FIXING: impossible task-event combinations
    # ========================================
    n_fixed = 0
    task_valid_event_count: dict[str, int] = {}
    for i in model.i:
        occurrences = task_occurrences[i]
        min_event = min(pos for _, _, pos, _ in occurrences)
        max_event = max(n_max - (length - pos) for _, _, pos, length in occurrences)
        valid = 0
        for n in model.n:
            if n < min_event or n > max_event:
                active_disjuncts[i, n].binary_indicator_var.fix(0)
                inactive_disjuncts[i, n].binary_indicator_var.fix(1)
                model.b[i, n].setub(0)
                n_fixed += 1
            else:
                valid += 1
        task_valid_event_count[i] = valid

    # Clamp task_imax to valid event count: if the event range has fewer slots
    # than imax, a higher imax is unreachable.
    n_clamped = 0
    for i in model.i:
        cur = pyo.value(model.task_imax[i])
        cap = task_valid_event_count.get(i, cur)
        if cap < cur:
            model.task_imax[i] = cap
            n_clamped += 1

    print(
        f"  [presolve] Fixed {n_fixed}/{len(model.i) * n_max} task-event disjuncts inactive."
        + (f" Clamped imax for {n_clamped} tasks." if n_clamped else "")
    )

    # Cross-event pool relations: for each unordered pair of distinct task-events
    # that can overlap, select one non-overlap order or one oriented overlap.
    def active_pool_task_events(pool_tasks_list):
        return [
            (i, n)
            for i in pool_tasks_list
            for n in model.n
            if not (
                active_disjuncts[i, n].binary_indicator_var.is_fixed()
                and pyo.value(active_disjuncts[i, n].binary_indicator_var) == 0
            )
        ]

    barr_te = active_pool_task_events(barr_tasks)
    jar_te = active_pool_task_events(jar_tasks)
    model._barr_pool_te = tuple(barr_te)
    model._jar_pool_te = tuple(jar_te)

    barr_count_ub = {(i, n): int(model.NumBarr[i, n].ub) for i, n in barr_te}
    jar_count_ub = {(i, n): int(model.NumJar[i, n].ub) for i, n in jar_te}

    # A strict rank resolves simultaneous starts consistently across all pairs.
    if not perf["barrique_flow_only"]:
        model.BarrStartRank = pyo.Var(
            barr_te,
            domain=pyo.NonNegativeReals,
            bounds=(0, max(0, len(barr_te) - 1)),
        )
    model.JarStartRank = pyo.Var(
        jar_te,
        domain=pyo.NonNegativeReals,
        bounds=(0, max(0, len(jar_te) - 1)),
    )

    def build_pool_pairs(pool_te, label):
        # Prune pairs whose ordering is already forced or whose conflict is impossible:
        #  (a) g06 chain on same task (i1 == i2: strict ordering across n)
        #  (b) ES/LF windows forcing strict precedence (LF[i1] <= ES[i2] etc.)
        pairs = []
        skip_same_i = skip_window = 0
        for (i1, n1), (i2, n2) in combinations(pool_te, 2):
            if i1 == i2:
                skip_same_i += 1
                continue
            if task_LF[i1] <= task_ES[i2] or task_LF[i2] <= task_ES[i1]:
                skip_window += 1
                continue
            pairs.append((i1, n1, i2, n2))
        kept = len(pairs)
        total = kept + skip_same_i + skip_window
        print(
            f"  [pool seq] {label} pairs: kept {kept}/{total} "
            f"(pruned: same-task {skip_same_i}, window {skip_window})"
        )
        return pairs

    barr_cross = (
        [] if perf["barrique_flow_only"] else build_pool_pairs(barr_te, "Barrique")
    )
    jar_cross = build_pool_pairs(jar_te, "Jar")

    # A barrique is either used for the first time or transferred directly from
    # a completed aging task. A transfer is possible only when the receiving
    # task starts within 24 h of the source task's finish.
    barr_reuse_arcs = []
    for i1, n1 in barr_te:
        for i2, n2 in barr_te:
            if (i1, n1) == (i2, n2):
                continue
            if i1 == i2 and n1 >= n2:
                continue
            earliest_source_finish = task_ES[i1] + pyo.value(model.alpha[i1])
            latest_destination_start = task_LF[i2] - pyo.value(model.alpha[i2])
            earliest_handoff = max(task_ES[i2], earliest_source_finish)
            latest_handoff = min(latest_destination_start, task_LF[i1] + 24.0)
            if earliest_handoff > latest_handoff:
                continue
            barr_reuse_arcs.append((i1, n1, i2, n2))
    model.BarrReuseArc = pyo.Set(initialize=barr_reuse_arcs, dimen=4)

    def _fresh_bounds(m, i, n):
        ub = barr_count_ub[i, n] if perf["tight_pool_bounds"] else n_barr
        return 0, ub

    def _reuse_bounds(m, i1, n1, i2, n2):
        ub = (
            min(barr_count_ub[i1, n1], barr_count_ub[i2, n2])
            if perf["tight_pool_bounds"]
            else n_barr
        )
        return 0, ub

    if perf["individual_barriques"]:
        model.BarrVesselUse = pyo.Var(model.JBAR, barr_te, domain=pyo.Binary)
        model.BarrVesselFresh = pyo.Var(model.JBAR, barr_te, domain=pyo.Binary)
        model.BarrVesselReuse = pyo.Var(
            model.JBAR, model.BarrReuseArc, domain=pyo.Binary
        )
    else:
        model.BarrFresh = pyo.Var(
            barr_te, domain=pyo.NonNegativeIntegers, bounds=_fresh_bounds
        )
        model.BarrReuse = pyo.Var(
            model.BarrReuseArc, domain=pyo.NonNegativeIntegers, bounds=_reuse_bounds
        )
        model.BarrReuseOn = pyo.Var(model.BarrReuseArc, domain=pyo.Binary)

    if perf["individual_jars"]:
        model.JarVesselUse = pyo.Var(model.JJAR, jar_te, domain=pyo.Binary)
        model.JarVesselUsed = pyo.Var(model.JJAR, domain=pyo.Binary)
    print(f"  [barrique reuse] Added {len(barr_reuse_arcs)} directed reuse arcs.")
    if not _hull:
        if barr_cross:
            model.BarrSeqFwd = pyo.Var(barr_cross, domain=pyo.Binary)
            model.BarrSeqRev = pyo.Var(barr_cross, domain=pyo.Binary)
            model.BarrOverlapFwd = pyo.Var(barr_cross, domain=pyo.Binary)
            model.BarrOverlapRev = pyo.Var(barr_cross, domain=pyo.Binary)
        if jar_cross:
            if perf["compact_jar_relations"]:
                # Codes: 00 sequence forward, 01 sequence reverse,
                # 10 overlap forward, 11 overlap reverse.
                model.JarRelOverlap = pyo.Var(jar_cross, domain=pyo.Binary)
                model.JarRelReverse = pyo.Var(jar_cross, domain=pyo.Binary)
            else:
                model.JarSeqFwd = pyo.Var(jar_cross, domain=pyo.Binary)
                model.JarSeqRev = pyo.Var(jar_cross, domain=pyo.Binary)
                model.JarOverlapFwd = pyo.Var(jar_cross, domain=pyo.Binary)
                model.JarOverlapRev = pyo.Var(jar_cross, domain=pyo.Binary)

    # ========================================
    # CONSTRAINTS (Standard Algebraic)
    # ========================================

    si_set = set(model.SI)

    # Material balances (h04, h03)
    def h04_rule(model, s, n):
        consumed = sum(
            model.rhoIScons[i, s] * model.b[i, n]
            for i in model.i
            if (i, s) in model.ICS
        )
        discard = model.Discard[s, 1] if s in si_set else 0.0
        return model.ST[s, n] == model.ST0[s] + consumed - discard

    model.h04 = pyo.Constraint(model.s, [1], rule=h04_rule)

    def h03_rule(model, s, n):
        produced = sum(
            model.rhoISprod[i, s] * model.b[i, n - 1]
            for i in model.i
            if (i, s) in model.IPS
        )
        consumed = sum(
            model.rhoIScons[i, s] * model.b[i, n]
            for i in model.i
            if (i, s) in model.ICS
        )
        discard = model.Discard[s, n] if s in si_set else 0.0
        return model.ST[s, n] == model.ST[s, n - 1] + produced + consumed - discard

    model.h03 = pyo.Constraint(model.s, [n for n in model.n if n > 1], rule=h03_rule)

    # Task sequencing g06 (Always active)
    def g06_rule(model, i, n):
        return model.Ts[i, n + 1] >= model.Tf[i, n]

    model.g06 = pyo.Constraint(
        model.i, [n for n in model.n if n < n_max], rule=g06_rule
    )

    # Unit capacity h09
    def h09_rule(model, j):
        if j in barr_set or j in jar_set:
            return pyo.Constraint.Skip
        terms = [
            model.alpha[i] * model.y[i, j, n] + model.beta[i] * model.bj[i, j, n]
            for i in model.inst
            for n in model.n
            if (i, j) in model.ij_nonpool
        ]
        if not terms:
            return pyo.Constraint.Feasible
        return sum(terms) <= model.MS

    model.h09 = pyo.Constraint(model.j, rule=h09_rule)

    # Zero Inventory h14/15/16
    def h14_rule(model, s, n):
        return model.ST[s, n] == 0

    model.h14 = pyo.Constraint(model.SZW, model.n, rule=h14_rule)

    def h15_rule(model, s, n):
        return model.ST[s, n] == 0

    model.h15 = pyo.Constraint(model.SNIS, model.n, rule=h15_rule)

    def h16_rule(model, s):
        produced = sum(
            model.rhoISprod[i, s] * model.b[i, n_max]
            for i in model.i
            if (i, s) in model.IPS
        )
        return model.ST[s, n_max] + produced == 0

    # A zero-wait state cannot carry material beyond the horizon.
    model.h16 = pyo.Constraint(model.SZW, rule=h16_rule)

    model.g17 = pyo.Constraint(
        model.SMarket, rule=lambda m, s: m.FinalProd[s] + m.Outsource[s] >= m.D[s]
    )
    model.g17_no_outsource = pyo.Constraint(
        [s for s in model.SMarket if pyo.value(model.OutsourceAllowed[s]) == 0],
        rule=lambda m, s: m.Outsource[s] == 0,
    )

    # A01/A02/A01st/A02st: unit-count bounds
    inst_nonpool = [i for i in model.inst if i not in pool_tasks]

    def A01_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij_nonpool)
            <= model.task_jmax[i] * active_disjuncts[i, n].binary_indicator_var
        )

    model.A01 = pyo.Constraint(inst_nonpool, model.n, rule=A01_rule)

    def A02_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij_nonpool)
            >= model.task_jmin[i] * active_disjuncts[i, n].binary_indicator_var
        )

    model.A02 = pyo.Constraint(inst_nonpool, model.n, rule=A02_rule)

    # Pool vessel-count bounds: active -> min_units <= count <= pool_size.
    model.BarrMin = pyo.Constraint(
        model.iBAR,
        model.n,
        rule=lambda m, i, n: m.NumBarr[i, n]
        >= m.task_jmin[i] * active_disjuncts[i, n].binary_indicator_var,
    )
    model.BarrMaxActive = pyo.Constraint(
        model.iBAR,
        model.n,
        rule=lambda m, i, n: m.NumBarr[i, n]
        <= (m.NumBarr[i, n].ub if perf["tight_pool_bounds"] else n_barr)
        * active_disjuncts[i, n].binary_indicator_var,
    )
    model.JarMin = pyo.Constraint(
        model.iJAR,
        model.n,
        rule=lambda m, i, n: m.NumJar[i, n]
        >= m.task_jmin[i] * active_disjuncts[i, n].binary_indicator_var,
    )
    model.JarMaxActive = pyo.Constraint(
        model.iJAR,
        model.n,
        rule=lambda m, i, n: m.NumJar[i, n]
        <= (m.NumJar[i, n].ub if perf["tight_pool_bounds"] else n_jar)
        * active_disjuncts[i, n].binary_indicator_var,
    )

    # A04: Total batch
    def A04_rule(model, i, n):
        if i in barr_tasks:
            return model.b[i, n] == barr_cap * model.NumBarr[i, n]
        if i in jar_tasks:
            return model.b[i, n] == jar_cap * model.NumJar[i, n]
        return model.b[i, n] == sum(
            model.bj[i, j, n] for j in model.j if (i, j) in model.ij_nonpool
        )

    model.A04 = pyo.Constraint(model.i, model.n, rule=A04_rule)

    # Event numbers are local task counters rather than common time intervals.
    if not perf["remove_event_index_capacity"]:
        model.BarrCapacity = pyo.Constraint(
            model.n,
            rule=lambda m, n: sum(m.NumBarr[i, n] for i in m.iBAR) <= n_barr,
        )
        model.JarCapacity = pyo.Constraint(
            model.n,
            rule=lambda m, n: sum(m.NumJar[i, n] for i in m.iJAR) <= n_jar,
        )

    # Every barrique task receives each vessel either as a first use (Fresh) or
    # as a timely transfer from a completed task.
    barr_incoming = {
        (i, n): [arc for arc in barr_reuse_arcs if arc[2:] == (i, n)]
        for i, n in barr_te
    }
    barr_outgoing = {
        (i, n): [arc for arc in barr_reuse_arcs if arc[:2] == (i, n)]
        for i, n in barr_te
    }
    if perf["individual_barriques"]:
        model.BarrVesselCount = pyo.Constraint(
            barr_te,
            rule=lambda m, i, n: m.NumBarr[i, n]
            == sum(m.BarrVesselUse[j, i, n] for j in m.JBAR),
        )
        model.BarrVesselBalance = pyo.Constraint(
            model.JBAR,
            barr_te,
            rule=lambda m, j, i, n: m.BarrVesselUse[j, i, n]
            == m.BarrVesselFresh[j, i, n]
            + sum(m.BarrVesselReuse[j, arc] for arc in barr_incoming[i, n]),
        )
        model.BarrVesselOutgoing = pyo.Constraint(
            model.JBAR,
            barr_te,
            rule=lambda m, j, i, n: sum(
                m.BarrVesselReuse[j, arc] for arc in barr_outgoing[i, n]
            )
            <= m.BarrVesselUse[j, i, n],
        )
        model.BarrVesselFirstUse = pyo.Constraint(
            model.JBAR,
            rule=lambda m, j: sum(m.BarrVesselFresh[j, i, n] for i, n in barr_te) <= 1,
        )
        ordered_barriques = sorted(model.JBAR, key=unit_order_key)
        model.BarrVesselSymmetry = pyo.ConstraintList()
        for previous, following in zip(ordered_barriques, ordered_barriques[1:]):
            model.BarrVesselSymmetry.add(
                sum(model.BarrVesselFresh[following, i, n] for i, n in barr_te)
                <= sum(model.BarrVesselFresh[previous, i, n] for i, n in barr_te)
            )
    else:
        model.BarrFreshBudget = pyo.Constraint(
            expr=sum(model.BarrFresh[i, n] for i, n in barr_te) <= n_barr
        )
        model.BarrFreshBalance = pyo.Constraint(
            barr_te,
            rule=lambda m, i, n: m.NumBarr[i, n]
            == m.BarrFresh[i, n] + sum(m.BarrReuse[arc] for arc in barr_incoming[i, n]),
        )
        model.BarrReuseOutgoing = pyo.Constraint(
            barr_te,
            rule=lambda m, i, n: sum(m.BarrReuse[arc] for arc in barr_outgoing[i, n])
            <= m.NumBarr[i, n],
        )
        model.BarrReuseLink = pyo.Constraint(
            model.BarrReuseArc,
            rule=lambda m, i1, n1, i2, n2: m.BarrReuse[i1, n1, i2, n2]
            <= (
                min(m.NumBarr[i1, n1].ub, m.NumBarr[i2, n2].ub)
                if perf["tight_pool_bounds"]
                else n_barr
            )
            * m.BarrReuseOn[i1, n1, i2, n2],
        )
        # Reuse flows are integral, so this makes the binary an exact indicator
        # of a positive transfer rather than merely an optional timing selector.
        model.BarrReuseIndicatorLower = pyo.Constraint(
            model.BarrReuseArc,
            rule=lambda m, i1, n1, i2, n2: m.BarrReuse[i1, n1, i2, n2]
            >= m.BarrReuseOn[i1, n1, i2, n2],
        )

    if perf["individual_jars"]:
        model.JarVesselCount = pyo.Constraint(
            jar_te,
            rule=lambda m, i, n: m.NumJar[i, n]
            == sum(m.JarVesselUse[j, i, n] for j in m.JJAR),
        )
        model.JarVesselUsedUpper = pyo.Constraint(
            model.JJAR,
            jar_te,
            rule=lambda m, j, i, n: m.JarVesselUse[j, i, n] <= m.JarVesselUsed[j],
        )
        model.JarVesselUsedLower = pyo.Constraint(
            model.JJAR,
            rule=lambda m, j: m.JarVesselUsed[j]
            <= sum(m.JarVesselUse[j, i, n] for i, n in jar_te),
        )
        ordered_jars = sorted(model.JJAR, key=unit_order_key)
        model.JarVesselSymmetry = pyo.ConstraintList()
        for previous, following in zip(ordered_jars, ordered_jars[1:]):
            model.JarVesselSymmetry.add(
                model.JarVesselUsed[following] <= model.JarVesselUsed[previous]
            )

    # A06: Max activations (non-storage tasks)
    model.A06 = pyo.Constraint(
        model.inst,
        rule=lambda m, i: sum(active_disjuncts[i, n].binary_indicator_var for n in m.n)
        <= m.task_imax[i],
    )
    aging_path_tasks = {}
    for ln in stg_int_lines:
        aging_step = next(
            step for step in lines_cfg[ln]["steps"] if is_aging_stage(step)
        )
        aging_path_tasks[ln] = (f"{aging_step}{ln}", f"{aging_step}Stg{ln}")
    model.AgingPathMax = pyo.Constraint(
        list(aging_path_tasks),
        rule=lambda m, ln: sum(
            active_disjuncts[task, n].binary_indicator_var
            for task in aging_path_tasks[ln]
            for n in m.n
        )
        <= m.iMax,
    )

    # A06_min: Min activations valid cuts for non-outsourceable lines
    min_act_indices = []
    min_act_rhs: dict[str, int] = {}
    aging_path_min_rhs: dict[str, int] = {}

    non_outsource_lines = [
        ln
        for ln in lines
        if line_product[ln] in model.SMarket
        and pyo.value(model.OutsourceAllowed[line_product[ln]]) == 0
    ]

    for line in non_outsource_lines:
        s = line_product[line]
        demand = pyo.value(model.D[s])
        if demand <= 0:
            continue
        steps = lines_cfg[line]["steps"]
        for step in steps:
            if step == "Pr":
                continue
            if line in stg_int_lines and step == "Stg":
                # Intermediate storage is an optional route.
                continue
            task_name = f"{step}{line}"
            if task_name not in model.i:
                continue
            if task_name in pool_tasks:
                if task_name in barr_tasks:
                    max_bmax = n_barr * barr_cap
                else:
                    max_bmax = n_jar * jar_cap
            else:
                max_bmax = max(
                    (
                        pyo.value(model.Bmax[task_name, j])
                        for j in model.j
                        if (task_name, j) in model.ij
                    ),
                    default=0.0,
                )
            if max_bmax <= 0:
                continue
            rhs = math.ceil(demand / max_bmax)
            if (
                line in young_lines
                and line_product[line] in dual_late_products_set
                and len(young_lines_by_product[line_product[line]]) == 1
            ):
                rhs = max(rhs, imax_young)
            rhs = min(
                rhs,
                task_valid_event_count.get(task_name, rhs),
                int(pyo.value(model.task_imax[task_name])),
            )
            if line in stg_int_lines and is_aging_stage(step):
                if rhs >= 2:
                    aging_path_min_rhs[line] = max(aging_path_min_rhs.get(line, 0), rhs)
                continue
            if rhs >= 2:
                cur = min_act_rhs.get(task_name, 0)
                if rhs > cur:
                    min_act_rhs[task_name] = rhs
                    if task_name not in min_act_indices:
                        min_act_indices.append(task_name)

    # A06_min_Pr for non-outsource pressed wine lines
    non_outsource_pressed = [
        ln
        for ln in pressing_lines_pr
        if line_product[ln] in model.SMarket
        and pyo.value(model.OutsourceAllowed[line_product[ln]]) == 0
    ]
    if "Pr" in model.i and non_outsource_pressed:
        rho_m_pr = (
            params.get("rho", {}).get("Pr", {}).get("rhoISprod", {}).get("m", 0.0)
        )
        max_b_pr = max(
            (pyo.value(model.Bmax["Pr", j]) for j in model.j if ("Pr", j) in model.ij),
            default=0.0,
        )
        if max_b_pr > 0 and rho_m_pr > 0:
            total_min_must = 0.0
            for ln in non_outsource_pressed:
                demand_ln = pyo.value(model.D[line_product[ln]])
                chain_yield = 1.0
                for step in lines_cfg[ln]["steps"]:
                    if step == "Pr":
                        continue
                    rho_entry = params.get("rho", {}).get(f"{step}{ln}", {})
                    for val in rho_entry.get("rhoISprod", {}).values():
                        if val < 1.0:
                            chain_yield *= val
                total_min_must += demand_ln / chain_yield
            min_pr_runs = math.ceil(total_min_must / (max_b_pr * rho_m_pr))
            min_pr_runs = min(
                min_pr_runs,
                task_valid_event_count.get("Pr", min_pr_runs),
                int(pyo.value(model.task_imax["Pr"])),
            )
            if min_pr_runs >= 2:
                model.A06_min_Pr = pyo.Constraint(
                    expr=sum(
                        active_disjuncts["Pr", n].binary_indicator_var for n in model.n
                    )
                    >= min_pr_runs
                )
                print(f"  [A06_min_Pr] Pr activations >= {min_pr_runs}")

    if min_act_indices:
        model.A06_min = pyo.Constraint(
            min_act_indices,
            rule=lambda m, i: sum(
                active_disjuncts[i, n].binary_indicator_var for n in m.n
            )
            >= min_act_rhs[i],
        )
        print(
            f"  [A06_min] Added min-activation cuts for {len(min_act_indices)} tasks: "
            + ", ".join(f"{i}>={min_act_rhs[i]}" for i in min_act_indices)
        )
    if aging_path_min_rhs:
        model.AgingPathMin = pyo.Constraint(
            list(aging_path_min_rhs),
            rule=lambda m, ln: sum(
                active_disjuncts[task, n].binary_indicator_var
                for task in aging_path_tasks[ln]
                for n in m.n
            )
            >= aging_path_min_rhs[ln],
        )
    # A09/A10: Unit event sequencing (non-pool units only)
    model.A09 = pyo.Constraint(
        j_nonpool,
        [n for n in model.n if n < n_max],
        rule=lambda m, j, n: m.Tsj[j, n + 1] >= m.Tfj[j, n],
    )
    model.A10 = pyo.Constraint(
        j_nonpool, model.n, rule=lambda m, j, n: m.Tfj[j, n] >= m.Tsj[j, n]
    )

    # Persistence
    def A13a_rule(model, i, ip, n):
        return (
            active_disjuncts[ip, n + 1].binary_indicator_var
            == active_disjuncts[i, n].binary_indicator_var
        )

    model.A13a = pyo.Constraint(
        model.tc1, [n for n in model.n if n < n_max], rule=A13a_rule
    )

    # Ecobulk Stg max duration: 720 h (1 month). Big-M relaxed when y[i,j,n]=0.
    # Tight M = Dmax_i - 720 (Dmax_i = LF_i - ES_i): an active Stg task's duration
    # never exceeds Dmax_i (duration_stor_ub), so 720 + M = Dmax_i is non-binding
    # at y=0.
    all_stg = list(model.iStgInt)
    eco_stg_index = [
        (i, j, n)
        for i in all_stg
        for j in model.JECO
        for n in model.n
        if (i, j) in model.ij_nonpool
    ]
    eco_M = {
        i: max(0.0, task_LF.get(i, H_val) - task_ES.get(i, 0.0) - 720.0)
        for i in all_stg
    }
    # "Assigned to an Ecobulk unit (y=1)" implies hold <= 720 h.
    if _hull:
        for i, j, n in eco_stg_index:
            guarded_relation(
                f"ecohold_{i}_{j}_{n}",
                model.Tf[i, n] - model.Ts[i, n] <= 720,
                [model.y[i, j, n]],
            )
    else:
        model.A_eco_stg_max = pyo.Constraint(
            eco_stg_index,
            rule=lambda m, i, j, n: m.Tf[i, n] - m.Ts[i, n]
            <= 720 + eco_M[i] * (1 - m.y[i, j, n]),
        )

    # Ends
    model.A18 = pyo.Constraint(
        model.SP,
        [n_max],
        rule=lambda m, s, n: m.ST[s, n]
        + sum(m.rhoISprod[i, s] * m.b[i, n] for i in m.i if (i, s) in m.IPS)
        == m.FinalProd[s],
    )

    # If a product has a required aging stage, all in-house final production of that
    # product must go through an aging task (direct or Stg-variant).
    model.A18_aging_link = pyo.Constraint(
        [p for p in model.SMarket if p in aging_tasks_by_product],
        rule=lambda m, p: m.FinalProd[p]
        <= sum(m.b[t, n] for t in aging_tasks_by_product[p] for n in m.n),
    )
    model.A21 = pyo.Constraint(
        model.ipst, [n_max], rule=lambda m, i, n: m.Tf[i, n] <= m.MS
    )

    # Products with an aging stage are not subject to the lateness penalty
    aging_products = set(aging_task_by_product.keys())

    model.LateDefByProduct = pyo.ConstraintList()
    campaign_branch = {}
    campaign_last_task = {}
    for line in lines:
        product = line_product[line]
        if product in aging_products:
            continue
        i_last = line_lateness_task[line]
        n_camps = imax_young if product in dual_late_products_set else 1

        ns_all = list(model.n)
        for k_idx, n in enumerate(ns_all):
            w_n = active_disjuncts[i_last, n].binary_indicator_var
            if w_n.is_fixed() and pyo.value(w_n) == 0:
                continue

            d_active = active_disjuncts[i_last, n]

            if n_camps == 1:
                # Single deadline: Big-M constraint
                M_late = task_LF[i_last] - deadlines_val[0]
                model.LateDefByProduct.add(
                    model.Tf[i_last, n]
                    <= deadlines_val[0] + model.Late[product, 1] + M_late * (1 - w_n)
                )
            else:
                # N-way GDP disjunction nested inside the active disjunct.
                # Each branch assigns the finish time to the deadline of campaign k.
                camp_disjuncts_list = []
                for camp_k in range(1, n_camps + 1):
                    dk = gdp.Disjunct()
                    dk.add_component(
                        "late_def",
                        pyo.Constraint(
                            expr=model.Tf[i_last, n]
                            <= model.DeadlineK[camp_k] + model.Late[product, camp_k]
                        ),
                    )
                    d_active.add_component(f"d_camp{camp_k}_late", dk)
                    camp_disjuncts_list.append(dk)
                    campaign_branch[line, n, camp_k] = dk.binary_indicator_var
                    campaign_last_task[line] = i_last
                d_active.add_component(
                    "Disj_campaign_late",
                    gdp.Disjunction(expr=camp_disjuncts_list),
                )

                active_prior = [
                    np
                    for np in ns_all[:k_idx]
                    if not (
                        active_disjuncts[i_last, np].binary_indicator_var.is_fixed()
                        and pyo.value(active_disjuncts[i_last, np].binary_indicator_var)
                        == 0
                    )
                ]
                # Fix campaigns that are impossible (not enough prior events).
                max_camp = min(len(active_prior) + 1, n_camps)
                for dk in camp_disjuncts_list[max_camp:]:
                    dk.binary_indicator_var.fix(0)

                if active_prior:
                    # Ordering equality: sum_{k=2}^{max_camp} (k-1)*c_k = S
                    # Exactly encodes "campaign = 1 + number of active prior events".
                    S = sum(
                        active_disjuncts[i_last, np].binary_indicator_var
                        for np in active_prior
                    )
                    d_active.add_component(
                        "CampOrderEq",
                        pyo.Constraint(
                            expr=sum(
                                (camp_k - 1)
                                * camp_disjuncts_list[camp_k - 1].binary_indicator_var
                                for camp_k in range(2, max_camp + 1)
                            )
                            == S
                        ),
                    )

    # Allocate every young-wine completion to its ordered campaign.
    campaign_batch_index = list(campaign_branch)
    campaign_event_index = sorted({(ln, n) for ln, n, _k in campaign_batch_index})
    campaign_line_index = sorted(
        {(ln, k) for ln, _n, k in campaign_batch_index},
        key=lambda item: (str(item[0]), int(item[1])),
    )
    model.CampaignBatch = pyo.Var(campaign_batch_index, domain=pyo.NonNegativeReals)
    model.CampaignBatchUpper = pyo.Constraint(
        campaign_batch_index,
        rule=lambda m, ln, n, k: m.CampaignBatch[ln, n, k]
        <= m.b[campaign_last_task[ln], n],
    )
    model.CampaignBatchIndicatorUpper = pyo.Constraint(
        campaign_batch_index,
        rule=lambda m, ln, n, k: m.CampaignBatch[ln, n, k]
        <= b_ub[campaign_last_task[ln]] * campaign_branch[ln, n, k],
    )
    model.CampaignBatchIndicatorLower = pyo.Constraint(
        campaign_batch_index,
        rule=lambda m, ln, n, k: m.CampaignBatch[ln, n, k]
        >= m.b[campaign_last_task[ln], n]
        - b_ub[campaign_last_task[ln]] * (1 - campaign_branch[ln, n, k]),
    )
    # The nested GDP already implies these identities at integer points.
    # We state them globally to expose the ordered-assignment structure to the root LP.
    model.CampaignBranchRow = pyo.Constraint(
        campaign_event_index,
        rule=lambda m, ln, n: sum(
            campaign_branch[ln, n, k]
            for _ln, _n, k in campaign_batch_index
            if _ln == ln and _n == n
        )
        == active_disjuncts[campaign_last_task[ln], n].binary_indicator_var,
    )
    model.CampaignBranchColumn = pyo.Constraint(
        campaign_line_index,
        rule=lambda m, ln, k: sum(
            campaign_branch[ln, n, k]
            for _ln, n, _k in campaign_batch_index
            if _ln == ln and _k == k
        )
        <= 1,
    )
    model.CampaignBatchConservation = pyo.Constraint(
        campaign_event_index,
        rule=lambda m, ln, n: sum(
            m.CampaignBatch[ln, n, k]
            for _ln, _n, k in campaign_batch_index
            if _ln == ln and _n == n
        )
        == m.b[campaign_last_task[ln], n],
    )

    campaign_lines_by_product = {
        s: [ln for ln in lines if line_product[ln] == s] for s in campaign_products
    }
    model._campaign_branch = campaign_branch
    model._campaign_last_task = campaign_last_task
    model._campaign_lines_by_product = campaign_lines_by_product
    model.CampaignProdDef = pyo.Constraint(
        campaign_index,
        rule=lambda m, s, k: m.CampaignProd[s, k]
        == sum(
            pyo.value(m.rhoISprod[campaign_last_task[ln], s])
            * m.CampaignBatch[ln, n, k]
            for ln in campaign_lines_by_product[s]
            for n in m.n
            if (ln, n, k) in campaign_branch
        ),
    )
    model.CampaignDemand = pyo.Constraint(
        campaign_index,
        rule=lambda m, s, k: m.CampaignProd[s, k] + m.CampaignOutsource[s, k]
        >= float(params["products"][s]["demand"]),
    )
    # Hull presence cut: without an assigned completion a campaign must
    # be fully outsourced.
    model.CampaignPresenceOutsource = pyo.Constraint(
        campaign_index,
        rule=lambda m, s, k: m.CampaignOutsource[s, k]
        >= float(params["products"][s]["demand"])
        * (
            1
            - sum(
                campaign_branch[ln, n, k]
                for ln in campaign_lines_by_product[s]
                for n in m.n
                if (ln, n, k) in campaign_branch
            )
        ),
    )
    required_campaign_lines = [
        ln
        for ln in campaign_last_task
        if pyo.value(model.OutsourceAllowed[line_product[ln]]) == 0
        and len(campaign_lines_by_product[line_product[ln]]) == 1
    ]
    model.CampaignRequiredActivations = pyo.Constraint(
        required_campaign_lines,
        rule=lambda m, ln: sum(
            active_disjuncts[campaign_last_task[ln], n].binary_indicator_var
            for n in m.n
        )
        >= imax_young,
    )
    model.CampaignTotalProd = pyo.Constraint(
        campaign_products,
        rule=lambda m, s: m.FinalProd[s]
        == sum(m.CampaignProd[s, k] for k in range(1, imax_young + 1)),
    )
    model.CampaignTotalOutsource = pyo.Constraint(
        campaign_products,
        rule=lambda m, s: m.Outsource[s]
        == sum(m.CampaignOutsource[s, k] for k in range(1, imax_young + 1)),
    )
    model.CampaignLateOn = pyo.Constraint(
        campaign_index,
        rule=lambda m, s, k: m.Late[s, k]
        <= max(0.0, H_val - deadlines_val[k - 1])
        * sum(
            campaign_branch[ln, n, k]
            for ln in campaign_lines_by_product[s]
            for n in m.n
            if (ln, n, k) in campaign_branch
        ),
    )

    # ========================================
    # DISJUNCTIONS (Combined Task-Unit, Eq. 2.1)
    # ========================================

    def add_constr(block, name, expr):
        block.add_component(name, pyo.Constraint(expr=expr))

    # assign_disjuncts stores unit-selection sub-disjuncts (used in LogicalConstraints)
    assign_disjuncts = {}

    for i in model.i:
        # Non-pool tasks have explicit unit-selection disjunctions
        # Pool tasks (barriques/jars) use integer count vars so we don't need nested disjuncts needed.
        compatible_units = (
            []
            if i in pool_tasks
            else [j for j in model.j if (i, j) in model.ij_nonpool]
        )
        for n in model.n:
            # Retrieve pre-created disjuncts, binary_indicator_var IS W_{i,n}
            d_active = active_disjuncts[i, n]

            if not compatible_units:
                add_constr(d_active, "nonempty", model.b[i, n] >= 0)

            # Nested unit selection: ⋁_{j in J_i}, skipped for pool tasks
            for j in compatible_units:
                d_assign = gdp.Disjunct()
                d_skip = gdp.Disjunct()
                d_active.add_component(f"d_assign_{j}", d_assign)
                d_active.add_component(f"d_skip_{j}", d_skip)
                assign_disjuncts[i, j, n] = d_assign

                # Assign: y=1, batch limits, time sync
                add_constr(d_assign, "y_on", model.y[i, j, n] == 1)
                bmin_val = float(pyo.value(model.Bmin[i, j]))
                bmax_val = float(pyo.value(model.Bmax[i, j]))
                if bmin_val == bmax_val:
                    # Fixed-capacity unit
                    add_constr(d_assign, "bfixed", model.bj[i, j, n] == bmin_val)
                else:
                    add_constr(d_assign, "bmin", model.bj[i, j, n] >= bmin_val)
                    add_constr(d_assign, "bmax", model.bj[i, j, n] <= bmax_val)
                add_constr(d_assign, "sync_ts", model.Tsj[j, n] == model.Ts[i, n])
                add_constr(d_assign, "sync_tf", model.Tfj[j, n] == model.Tf[i, n])

                # Skip: y=0, bj=0
                add_constr(d_skip, "y_off", model.y[i, j, n] == 0)
                add_constr(d_skip, "bj_zero", model.bj[i, j, n] == 0)

                d_active.add_component(
                    f"Disj_unit_{j}", gdp.Disjunction(expr=[d_assign, d_skip])
                )

            # Inactive Disjunct (not W_{i,n})
            d_inactive = inactive_disjuncts[i, n]

            if compatible_units:
                d_inactive.y_zero = pyo.ConstraintList()
                d_inactive.bj_zero = pyo.ConstraintList()
                for j in compatible_units:
                    d_inactive.y_zero.add(model.y[i, j, n] == 0)
                    d_inactive.bj_zero.add(model.bj[i, j, n] == 0)
            else:
                add_constr(d_inactive, "nonempty", model.b[i, n] >= 0)

            # Outer disjunction
            model.add_component(
                f"Disj_Task_{i}_{n}",
                gdp.Disjunction(expr=[d_active, d_inactive]),
            )

    # ========================================
    # GLOBAL DURATION LINK (replaces per-disjunct duration; no M=H Big-M)
    # ========================================
    # Processing tasks (incl. pool): Tf = Ts + alpha*W + beta*b.
    #   W=1 -> Tf = Ts + alpha + beta*b (active duration)
    #   W=0 -> b=0 (b_on_off) -> Tf = Ts (idle collapse)
    # Storage tasks (iStgInt, variable duration): Ts+alpha <= Tf <= Ts+Dmax when
    #   active, Tf = Ts when idle. Dmax = task time-window width.
    proc_tasks = [i for i in model.i if i not in model.iStgInt]
    stor_tasks = list(model.iStgInt)
    # Per-task time-window width Dmax = LF - ES
    task_Dmax = {
        i: max(0.0, task_LF.get(i, H_val) - task_ES.get(i, 0.0)) for i in model.i
    }

    model.duration_proc = pyo.Constraint(
        proc_tasks,
        model.n,
        rule=lambda m, i, n: m.Tf[i, n]
        == m.Ts[i, n]
        + m.alpha[i] * active_disjuncts[i, n].binary_indicator_var
        + m.beta[i] * m.b[i, n],
    )
    model.duration_stor_lb = pyo.Constraint(
        stor_tasks,
        model.n,
        rule=lambda m, i, n: m.Tf[i, n]
        >= m.Ts[i, n] + m.alpha[i] * active_disjuncts[i, n].binary_indicator_var,
    )
    model.duration_stor_ub = pyo.Constraint(
        stor_tasks,
        model.n,
        rule=lambda m, i, n: m.Tf[i, n]
        <= m.Ts[i, n] + task_Dmax[i] * active_disjuncts[i, n].binary_indicator_var,
    )

    # Idle-collapse (symmetry break), globalized with tight M = window width.
    # g06 already gives Ts[i,n] >= Tf[i,n-1]. This caps it from above:
    #   W=0 -> Ts[i,n] = Tf[i,n-1] (collapse idle event onto previous finish)
    #   W=1 -> relaxed by Dmax_i   (active task may start after the previous finish)
    model.idle_collapse = pyo.Constraint(
        model.i,
        [n for n in model.n if n > 1],
        rule=lambda m, i, n: m.Ts[i, n]
        <= m.Tf[i, n - 1] + task_Dmax[i] * active_disjuncts[i, n].binary_indicator_var,
    )

    # A03: At most one task per unit per event (non-pool units only)
    j_with_tasks = [
        j for j in j_nonpool if any((i, j) in model.ij_nonpool for i in model.i)
    ]
    model.A03 = pyo.Constraint(
        j_with_tasks,
        model.n,
        rule=lambda m, j, n: sum(m.y[i, j, n] for i in m.i if (i, j) in m.ij_nonpool)
        <= 1,
    )

    # ========================================
    # TASK-PAIR BIG-M COMPUTATION
    # ========================================
    # M[i_prod, i_cons] = LF[i_prod] - ES[i_cons]
    def M_ge(i_prod: str, i_cons: str) -> float:
        """Big-M for Ts[cons] >= Tf[prod] - M*(2-W_prod-W_cons)."""
        return max(0.0, task_LF[i_prod] - task_ES[i_cons])

    def M_le(i_prod: str, i_cons: str) -> float:
        """Big-M for Ts[cons] <= Tf[prod] + M*(2-W_prod-W_cons)."""
        return max(0.0, task_LF[i_cons] - task_ES[i_prod])

    def M_bar_ge(i_prev: str, i_next: str) -> float:
        """Big-M for handoff timing: Ts[next] >= Tf[prev] - M*(1-IsHandoff)."""
        return max(0.0, task_LF[i_prev] - task_ES[i_next])

    def M_reuse_window(i_prev: str, i_next: str) -> float:
        """Big-M for Ts[next] <= Tf[prev] + 24 when a reuse arc is active."""
        return max(0.0, task_LF[i_next] - task_ES[i_prev] - 24.0)

    # A positive reuse flow means that these particular barriques leave the
    # source task and are refilled by the destination task.
    if perf["individual_barriques"]:
        model.BarrVesselReuseAfter = pyo.Constraint(
            model.JBAR,
            model.BarrReuseArc,
            rule=lambda m, j, i1, n1, i2, n2: m.Tf[i1, n1]
            <= m.Ts[i2, n2]
            + M_bar_ge(i1, i2) * (1 - m.BarrVesselReuse[j, i1, n1, i2, n2]),
        )
        model.BarrVesselReuseWithin24h = pyo.Constraint(
            model.JBAR,
            model.BarrReuseArc,
            rule=lambda m, j, i1, n1, i2, n2: m.Ts[i2, n2]
            <= m.Tf[i1, n1]
            + 24.0
            + M_reuse_window(i1, i2) * (1 - m.BarrVesselReuse[j, i1, n1, i2, n2]),
        )
    elif _hull:
        for i1, n1, i2, n2 in model.BarrReuseArc:
            reuse_on = model.BarrReuseOn[i1, n1, i2, n2]
            guarded_relation(
                f"barrreuse_after_{i1}_{n1}_{i2}_{n2}",
                model.Tf[i1, n1] <= model.Ts[i2, n2],
                [reuse_on],
            )
            guarded_relation(
                f"barrreuse_24h_{i1}_{n1}_{i2}_{n2}",
                model.Ts[i2, n2] <= model.Tf[i1, n1] + 24.0,
                [reuse_on],
            )
    else:
        model.BarrReuseAfter = pyo.Constraint(
            model.BarrReuseArc,
            rule=lambda m, i1, n1, i2, n2: m.Tf[i1, n1]
            <= m.Ts[i2, n2] + M_bar_ge(i1, i2) * (1 - m.BarrReuseOn[i1, n1, i2, n2]),
        )
        model.BarrReuseWithin24h = pyo.Constraint(
            model.BarrReuseArc,
            rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
            <= m.Tf[i1, n1]
            + 24.0
            + M_reuse_window(i1, i2) * (1 - m.BarrReuseOn[i1, n1, i2, n2]),
        )

    # Pool cross-event relations:
    # two non-overlap orders and two oriented overlap branches.
    def _pool_manual(prefix, cross, fwd, rev, ov_fwd, ov_rev, rank):
        model.add_component(
            f"{prefix}SeqFwdCon",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Tf[i1, n1]
                <= m.Ts[i2, n2] + M_bar_ge(i1, i2) * (1 - fwd[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}SeqRevCon",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Tf[i2, n2]
                <= m.Ts[i1, n1] + M_bar_ge(i2, i1) * (1 - rev[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}OverlapFwdStartOrder",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i1, n1]
                <= m.Ts[i2, n2] + M_ge(i1, i2) * (1 - ov_fwd[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}OverlapFwdActive",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
                <= m.Tf[i1, n1] + M_le(i1, i2) * (1 - ov_fwd[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}OverlapRevStartOrder",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
                <= m.Ts[i1, n1] + M_ge(i2, i1) * (1 - ov_rev[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}OverlapRevActive",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i1, n1]
                <= m.Tf[i2, n2] + M_le(i2, i1) * (1 - ov_rev[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}RelationChoice",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: fwd[i1, n1, i2, n2]
                + rev[i1, n1, i2, n2]
                + ov_fwd[i1, n1, i2, n2]
                + ov_rev[i1, n1, i2, n2]
                == 1,
            ),
        )
        rank_M = max(1, len(rank))
        model.add_component(
            f"{prefix}RankFwd",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: rank[i1, n1] + 1
                <= rank[i2, n2]
                + rank_M * (1 - fwd[i1, n1, i2, n2] - ov_fwd[i1, n1, i2, n2]),
            ),
        )
        model.add_component(
            f"{prefix}RankRev",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: rank[i2, n2] + 1
                <= rank[i1, n1]
                + rank_M * (1 - rev[i1, n1, i2, n2] - ov_rev[i1, n1, i2, n2]),
            ),
        )

    def _pool_compact(prefix, cross, overlap, reverse, rank):
        """Exact two-bit encoding of the four interval-relation branches.

        ``overlap`` selects sequence (0) versus overlap (1), and ``reverse``
        selects forward (0) versus reverse (1). For binary vals exactly one
        code has zero Hamming distance, so the integer feasible set is the same
        as the four-way one-hot encoding.
        """

        def d00(key):
            return overlap[key] + reverse[key]

        def d01(key):
            return overlap[key] + (1 - reverse[key])

        def d10(key):
            return (1 - overlap[key]) + reverse[key]

        def d11(key):
            return (1 - overlap[key]) + (1 - reverse[key])

        model.add_component(
            f"{prefix}SeqFwdCon",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Tf[i1, n1]
                <= m.Ts[i2, n2] + M_bar_ge(i1, i2) * d00((i1, n1, i2, n2)),
            ),
        )
        model.add_component(
            f"{prefix}SeqRevCon",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Tf[i2, n2]
                <= m.Ts[i1, n1] + M_bar_ge(i2, i1) * d01((i1, n1, i2, n2)),
            ),
        )
        model.add_component(
            f"{prefix}OverlapFwdStartOrder",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i1, n1]
                <= m.Ts[i2, n2] + M_ge(i1, i2) * d10((i1, n1, i2, n2)),
            ),
        )
        model.add_component(
            f"{prefix}OverlapFwdActive",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
                <= m.Tf[i1, n1] + M_le(i1, i2) * d10((i1, n1, i2, n2)),
            ),
        )
        model.add_component(
            f"{prefix}OverlapRevStartOrder",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
                <= m.Ts[i1, n1] + M_ge(i2, i1) * d11((i1, n1, i2, n2)),
            ),
        )
        model.add_component(
            f"{prefix}OverlapRevActive",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: m.Ts[i1, n1]
                <= m.Tf[i2, n2] + M_le(i2, i1) * d11((i1, n1, i2, n2)),
            ),
        )
        rank_M = max(1, len(rank))
        model.add_component(
            f"{prefix}RankFwd",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: rank[i1, n1] + 1
                <= rank[i2, n2] + rank_M * reverse[i1, n1, i2, n2],
            ),
        )
        model.add_component(
            f"{prefix}RankRev",
            pyo.Constraint(
                cross,
                rule=lambda m, i1, n1, i2, n2: rank[i2, n2] + 1
                <= rank[i1, n1] + rank_M * (1 - reverse[i1, n1, i2, n2]),
            ),
        )

    def _add_pool_cumulative_capacity(prefix, pool_te, cross, num, cap, ov_fwd, ov_rev):
        """Enforce capacity at every pool-task start.

        For a fixed set of intervals, cumulative usage can change only at a task
        start or finish. Checking every task start is sufficient.
        """
        use_index = []
        overlap_indicator = {}
        for i1, n1, i2, n2 in cross:
            # overlap-fwd: (i1,n1) is active at the start of (i2,n2)
            use_index.append((i1, n1, i2, n2))
            overlap_indicator[i1, n1, i2, n2] = ov_fwd[i1, n1, i2, n2]
            # overlap-rev: (i2,n2) is active at the start of (i1,n1)
            use_index.append((i2, n2, i1, n1))
            overlap_indicator[i2, n2, i1, n1] = ov_rev[i1, n1, i2, n2]

        index_name = f"{prefix}UseIndex"
        use_name = f"{prefix}UseAtStart"
        model.add_component(index_name, pyo.Set(initialize=use_index, dimen=4))
        use_index_set = getattr(model, index_name)

        def source_ub(ib, nb):
            return num[ib, nb].ub if perf["tight_pool_bounds"] else cap

        model.add_component(
            use_name,
            pyo.Var(
                use_index_set,
                domain=pyo.NonNegativeReals,
                bounds=lambda m, ib, nb, ia, na: (0, source_ub(ib, nb)),
            ),
        )
        use = getattr(model, use_name)
        model.add_component(
            f"{prefix}UseUpperCount",
            pyo.Constraint(
                use_index_set,
                rule=lambda m, ib, nb, ia, na: use[ib, nb, ia, na] <= num[ib, nb],
            ),
        )
        model.add_component(
            f"{prefix}UseUpperIndicator",
            pyo.Constraint(
                use_index_set,
                rule=lambda m, ib, nb, ia, na: use[ib, nb, ia, na]
                <= source_ub(ib, nb) * overlap_indicator[ib, nb, ia, na],
            ),
        )
        model.add_component(
            f"{prefix}UseLower",
            pyo.Constraint(
                use_index_set,
                rule=lambda m, ib, nb, ia, na: use[ib, nb, ia, na]
                >= num[ib, nb]
                - source_ub(ib, nb) * (1 - overlap_indicator[ib, nb, ia, na]),
            ),
        )

        by_checkpoint = {
            (i, n): [key for key in use_index if key[2:] == (i, n)] for i, n in pool_te
        }
        model.add_component(
            f"{prefix}CapacityAtStart",
            pyo.Constraint(
                pool_te,
                rule=lambda m, i, n: num[i, n]
                + sum(use[key] for key in by_checkpoint[i, n])
                <= cap,
            ),
        )

    def _add_pool_cumulative_capacity_compact(
        prefix, pool_te, cross, num, cap, overlap, reverse
    ):
        """Cumulative profile linearization for the exact two-bit relation code."""
        use_index = []
        code = {}
        for i1, n1, i2, n2 in cross:
            use_index.append((i1, n1, i2, n2))
            code[i1, n1, i2, n2] = (overlap[i1, n1, i2, n2], reverse[i1, n1, i2, n2], 0)
            use_index.append((i2, n2, i1, n1))
            code[i2, n2, i1, n1] = (overlap[i1, n1, i2, n2], reverse[i1, n1, i2, n2], 1)

        index_name = f"{prefix}UseIndex"
        use_name = f"{prefix}UseAtStart"
        model.add_component(index_name, pyo.Set(initialize=use_index, dimen=4))
        use_index_set = getattr(model, index_name)

        def source_ub(ib, nb):
            return num[ib, nb].ub if perf["tight_pool_bounds"] else cap

        model.add_component(
            use_name,
            pyo.Var(
                use_index_set,
                domain=pyo.NonNegativeReals,
                bounds=lambda m, ib, nb, ia, na: (0, source_ub(ib, nb)),
            ),
        )
        use = getattr(model, use_name)
        model.add_component(
            f"{prefix}UseUpperCount",
            pyo.Constraint(
                use_index_set,
                rule=lambda m, ib, nb, ia, na: use[ib, nb, ia, na] <= num[ib, nb],
            ),
        )
        model.add_component(f"{prefix}UseCode", pyo.ConstraintList())
        use_code = getattr(model, f"{prefix}UseCode")
        for key in use_index:
            ib, nb, _, _ = key
            ov, rev, reverse_value = code[key]
            ub = source_ub(ib, nb)
            use_code.add(use[key] <= ub * ov)
            if reverse_value:
                use_code.add(use[key] <= ub * rev)
                use_code.add(use[key] >= num[ib, nb] - ub * ((1 - ov) + (1 - rev)))
            else:
                use_code.add(use[key] <= ub * (1 - rev))
                use_code.add(use[key] >= num[ib, nb] - ub * ((1 - ov) + rev))

        by_checkpoint = {
            (i, n): [key for key in use_index if key[2:] == (i, n)] for i, n in pool_te
        }
        model.add_component(
            f"{prefix}CapacityAtStart",
            pyo.Constraint(
                pool_te,
                rule=lambda m, i, n: num[i, n]
                + sum(use[key] for key in by_checkpoint[i, n])
                <= cap,
            ),
        )

    def _add_named_jar_conflicts(overlap_fwd, overlap_rev=None):
        """Forbid one named jar from serving two intervals that overlap."""
        model.JarVesselOverlapConflict = pyo.ConstraintList()
        for key in jar_cross:
            i1, n1, i2, n2 = key
            overlap = overlap_fwd[key]
            if overlap_rev is not None:
                overlap += overlap_rev[key]
            for vessel in model.JJAR:
                model.JarVesselOverlapConflict.add(
                    model.JarVesselUse[vessel, i1, n1]
                    + model.JarVesselUse[vessel, i2, n2]
                    <= 2 - overlap
                )

    if _hull:
        barr_overlap_fwd = {}
        barr_overlap_rev = {}
        for i1, n1, i2, n2 in barr_cross:
            fwd, rev = pool_seq(
                f"barrseq_{i1}_{n1}_{i2}_{n2}",
                i1,
                n1,
                i2,
                n2,
                model.BarrStartRank,
            )
            barr_overlap_fwd[i1, n1, i2, n2] = fwd
            barr_overlap_rev[i1, n1, i2, n2] = rev
        jar_overlap_fwd = {}
        jar_overlap_rev = {}
        for i1, n1, i2, n2 in jar_cross:
            fwd, rev = pool_seq(
                f"jarseq_{i1}_{n1}_{i2}_{n2}",
                i1,
                n1,
                i2,
                n2,
                model.JarStartRank,
            )
            jar_overlap_fwd[i1, n1, i2, n2] = fwd
            jar_overlap_rev[i1, n1, i2, n2] = rev
        if barr_cross:
            _add_pool_cumulative_capacity(
                "Barr",
                barr_te,
                barr_cross,
                model.NumBarr,
                n_barr,
                barr_overlap_fwd,
                barr_overlap_rev,
            )
        if jar_cross:
            if perf["individual_jars"]:
                _add_named_jar_conflicts(jar_overlap_fwd, jar_overlap_rev)
            else:
                _add_pool_cumulative_capacity(
                    "Jar",
                    jar_te,
                    jar_cross,
                    model.NumJar,
                    n_jar,
                    jar_overlap_fwd,
                    jar_overlap_rev,
                )
    else:
        if barr_cross:
            _pool_manual(
                "Barr",
                barr_cross,
                model.BarrSeqFwd,
                model.BarrSeqRev,
                model.BarrOverlapFwd,
                model.BarrOverlapRev,
                model.BarrStartRank,
            )
            _add_pool_cumulative_capacity(
                "Barr",
                barr_te,
                barr_cross,
                model.NumBarr,
                n_barr,
                model.BarrOverlapFwd,
                model.BarrOverlapRev,
            )
        if jar_cross:
            if perf["compact_jar_relations"]:
                _pool_compact(
                    "Jar",
                    jar_cross,
                    model.JarRelOverlap,
                    model.JarRelReverse,
                    model.JarStartRank,
                )
                if perf["individual_jars"]:
                    _add_named_jar_conflicts(model.JarRelOverlap)
                else:
                    _add_pool_cumulative_capacity_compact(
                        "Jar",
                        jar_te,
                        jar_cross,
                        model.NumJar,
                        n_jar,
                        model.JarRelOverlap,
                        model.JarRelReverse,
                    )
            else:
                _pool_manual(
                    "Jar",
                    jar_cross,
                    model.JarSeqFwd,
                    model.JarSeqRev,
                    model.JarOverlapFwd,
                    model.JarOverlapRev,
                    model.JarStartRank,
                )
                if perf["individual_jars"]:
                    _add_named_jar_conflicts(model.JarOverlapFwd, model.JarOverlapRev)
                else:
                    _add_pool_cumulative_capacity(
                        "Jar",
                        jar_te,
                        jar_cross,
                        model.NumJar,
                        n_jar,
                        model.JarOverlapFwd,
                        model.JarOverlapRev,
                    )

    # ========================================
    # PRECEDENCE (Conditional Constraints, Eq. 2.2)
    # ========================================

    # Precedence: "i' active at n AND i active at n+1 implies Ts_{i,n+1} >= Tf_{i',n}"
    #   (and equality for zero-wait / NIS / Ecobulk pairs). In hull mode, this is a
    #   guarded GDP disjunction; in Big-M mode it is the structure-aware Big-M
    #   reformulation written directly on the shared activation indicators.
    zw_nis_eco_pairs = [
        (i_cons, i_prod, s)
        for i_cons, i_prod, s in precedence_pairs
        if s in model.SZW or s in model.SNIS or s in model.SFISEco
    ]
    zw_nis_eco_set = set(zw_nis_eco_pairs)

    if _hull:
        for i_cons, i_prod, s in precedence_pairs:
            for n in model.n:
                if n >= n_max:
                    continue
                guard = [
                    active_disjuncts[i_prod, n].binary_indicator_var,
                    active_disjuncts[i_cons, n + 1].binary_indicator_var,
                ]
                guarded_relation(
                    f"precge_{i_prod}_{i_cons}_{n}",
                    model.Ts[i_cons, n + 1] >= model.Tf[i_prod, n],
                    guard,
                )
                if (i_cons, i_prod, s) in zw_nis_eco_set:
                    guarded_relation(
                        f"precle_{i_prod}_{i_cons}_{n}",
                        model.Ts[i_cons, n + 1] <= model.Tf[i_prod, n],
                        guard,
                    )
    else:

        def _wprod(ip, n):
            return active_disjuncts[ip, n].binary_indicator_var

        def _wcons(ic, n):
            return active_disjuncts[ic, n].binary_indicator_var

        model.prec_ge = pyo.Constraint(
            [
                (ic, ip, s, n)
                for ic, ip, s in precedence_pairs
                for n in model.n
                if n < n_max
            ],
            rule=lambda m, ic, ip, s, n: m.Ts[ic, n + 1]
            >= m.Tf[ip, n] - M_ge(ip, ic) * (2 - _wprod(ip, n) - _wcons(ic, n + 1)),
        )
        model.prec_le = pyo.Constraint(
            [
                (ic, ip, s, n)
                for ic, ip, s in zw_nis_eco_pairs
                for n in model.n
                if n < n_max
            ],
            rule=lambda m, ic, ip, s, n: m.Ts[ic, n + 1]
            <= m.Tf[ip, n] + M_le(ip, ic) * (2 - _wprod(ip, n) - _wcons(ic, n + 1)),
        )

    # ========================================
    # OBJECTIVE FUNCTION
    # ========================================
    model.Revenue = pyo.Expression(
        expr=sum(model.Price[s] * model.D[s] for s in model.SMarket)
    )
    model.OutsourcingCost = pyo.Expression(
        expr=sum(model.CostOutsourcing[s] * model.Outsource[s] for s in model.SMarket)
    )
    lateness_terms = [
        model.Late[s, k]
        for k in range(1, imax_young + 1)
        for s in (all_late_products if k == 1 else dual_late_products)
    ]
    model.Lateness = pyo.Expression(expr=sum(lateness_terms) if lateness_terms else 0.0)
    model.LatenessCost = pyo.Expression(
        expr=model.penaltyLate * sum(lateness_terms) if lateness_terms else 0.0
    )
    model.RawMaterialCost = pyo.Expression(
        expr=sum(
            model.RawMaterialCostPerL[s] * (-model.rhoIScons[i, s]) * model.b[i, n]
            for s in model.SR
            for i in model.i
            for n in model.n
            if (i, s) in model.ICS
        )
    )
    model.GrapeSkinRevenue = pyo.Expression(
        expr=model.GrapeSkinValue * model.FinalProd["grape_skin"]
        if has_pressing
        else 0.0
    )

    # beta=0 enforced for exterior tasks, so cooling time = alpha[i].
    # Disjunctive bounds force bj[i,j,n] = 0 when y[i,j,n] = 0, so bj already
    # carries the on/off state. Cost = costCooling * alpha[i] * bj[i,j,n] is
    # exact without auxiliary McCormick variables.
    model.CoolingCost = pyo.Expression(
        expr=model.costCooling
        * sum(
            model.alpha[i] * model.bj[i, j, n]
            for (i, j) in model.ij_nonpool
            if j in model.JEXT
            for n in model.n
        )
    )

    model.DiscardCost = pyo.Expression(
        expr=model.penaltyDiscard
        * sum(model.Discard[s, n] for s in model.SI for n in model.n)
    )

    stg_task_names = [f"Stg{ln}" for ln in stg_int_lines if f"Stg{ln}" in set(model.i)]
    model.StgHoldingCost = pyo.Expression(
        expr=model.stgHoldingCost
        * sum(model.b[t, n] for t in stg_task_names for n in model.n)
        if stg_task_names
        else 0.0
    )

    model.MakespanPenalty = pyo.Expression(expr=model.penaltyMS * model.MS)
    # Start-time pulling is applied only in a second, lexicographic solve after
    # the primary MIP has terminated.
    model.TimePullPenalty = pyo.Expression(
        expr=1e-6 * sum(model.Ts[i, n] for i in model.i for n in model.n)
    )
    model.Profit = pyo.Expression(
        expr=model.Revenue
        + model.GrapeSkinRevenue
        - model.OutsourcingCost
        - model.RawMaterialCost
        - model.LatenessCost
        - model.DiscardCost
        - model.CoolingCost
    )

    model.PrimaryObjective = pyo.Expression(
        expr=model.GrapeSkinRevenue
        - model.OutsourcingCost
        - model.LatenessCost
        - model.RawMaterialCost
        - model.MakespanPenalty
        - model.CoolingCost
        - model.DiscardCost
        - model.StgHoldingCost
    )

    def obj_func(model):
        return model.PrimaryObjective

    model.OBJ = pyo.Objective(rule=obj_func, sense=pyo.maximize)

    # ========================================
    # LP TIGHTENING
    # ========================================
    # The helper must be called after y / Bmin / Bmax are fully
    # initialised and before the GDP transformation flattens the disjuncts.
    add_cover_cuts(model)

    # Global on/off link: b[i, n] <= b_ub[i] * W[i, n].
    # Exact (W=0 -> b=0; W=1 -> b <= b_ub). Tighter than cover_cut for pool tasks
    # too, since it ties the batch directly to the activation binary rather than
    # to the per-unit y / NumBarr / NumJar counts.
    def b_on_off_rule(m, i, n):
        return m.b[i, n] <= b_ub[i] * active_disjuncts[i, n].binary_indicator_var

    model.b_on_off = pyo.Constraint(model.i, model.n, rule=b_on_off_rule)
    print(
        f"  [b_on_off] Added {len(model.i) * len(model.n)} batch on/off "
        "linking constraints."
    )

    # Identical-unit symmetry breaking (valid global-count form).
    # Instances of a multi-quantity unit are interchangeable, so order them by
    # TOTAL usage over the whole schedule: sum_{i,n} y[i,u_{m+1},n] <= sum_{i,n}
    # y[i,u_m,n].
    ij_np_set = set(model.ij_nonpool)
    tasks_on_unit = {}
    for unit_inst in model.j:
        ts = [i for i in model.i if (i, unit_inst) in ij_np_set]
        if ts:
            tasks_on_unit[unit_inst] = ts

    sym_pairs = []
    for insts in unit_instances_by_base.values():
        used = [u for u in insts if u in tasks_on_unit]
        for m in range(len(used) - 1):
            sym_pairs.append((used[m], used[m + 1]))

    if sym_pairs:

        def _unit_usage(u):
            return sum(model.y[i, u, n] for i in tasks_on_unit[u] for n in model.n)

        model.sym_break = pyo.ConstraintList()
        for u_lo, u_hi in sym_pairs:
            model.sym_break.add(_unit_usage(u_hi) <= _unit_usage(u_lo))
        print(
            f"  [sym_break] Added {len(sym_pairs)} identical-unit usage-ordering "
            f"cuts across {len({u for p in sym_pairs for u in p})} instances."
        )

    # Stash for solver-side branching priority assignment.
    model._active_disjuncts = active_disjuncts

    # Two reformulations of the same disjunctive model:
    #  "bigm" -> gdp.bigm on the activation/assignment disjunctions, with the
    #             timing/precedence/pool logic already written as structure-aware
    #             Big-M (the delivered model),
    #  "hull" -> gdp.hull on the full disjunctive model (all logic as disjunctions).
    print(f"Applying GDP {transformation_type} transformation...")
    if transformation_type == "hull":
        pyo.TransformationFactory("gdp.hull").apply_to(model)
    else:
        pyo.TransformationFactory("gdp.bigm").apply_to(model)

    return model


def _has_solver_incumbent(results):
    """Return whether the solver reported at least one feasible solution."""
    try:
        count = results.problem.number_of_solutions
        if count is not None:
            return int(count) > 0
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return len(results.solution) > 0
    except (AttributeError, TypeError):
        return False


def _audit_pool_solution(model, tol=1e-5):
    """Validate and decompose the aggregate pool solution after every solve."""

    def integer_value(var, label):
        value = pyo.value(var, exception=False)
        if value is None or abs(value - round(value)) > tol:
            raise RuntimeError(f"Pool audit: {label} is not integral ({value}).")
        return int(round(value))

    fresh = getattr(model, "BarrFresh", None)
    reuse = getattr(model, "BarrReuse", None)
    if fresh is not None and reuse is not None:
        nodes = sorted(
            list(fresh),
            key=lambda key: (pyo.value(model.Ts[key]), str(key[0]), int(key[1])),
        )
        incoming_paths = {key: [] for key in nodes}
        paths = {}
        next_path = 0
        outgoing = {key: [] for key in nodes}
        for arc in model.BarrReuseArc:
            amount = integer_value(reuse[arc], f"BarrReuse{arc}")
            if amount:
                i1, n1, i2, n2 = arc
                finish = pyo.value(model.Tf[i1, n1])
                start = pyo.value(model.Ts[i2, n2])
                if finish > start + tol or start > finish + 24.0 + tol:
                    raise RuntimeError(
                        f"Pool audit: positive reuse {arc} violates the 24 h window."
                    )
                outgoing[i1, n1].append((start, arc, amount))

        for node in nodes:
            count = integer_value(model.NumBarr[node], f"NumBarr{node}")
            new_count = integer_value(fresh[node], f"BarrFresh{node}")
            held = list(incoming_paths[node])
            for _ in range(new_count):
                paths[next_path] = []
                held.append(next_path)
                next_path += 1
            if len(held) != count:
                raise RuntimeError(
                    f"Pool audit: barrique balance at {node} gives {len(held)} "
                    f"paths for a count of {count}."
                )
            for path in held:
                paths[path].append(node)
            cursor = 0
            for _, arc, amount in sorted(outgoing[node]):
                destination = arc[2:]
                selected = held[cursor : cursor + amount]
                if len(selected) != amount:
                    raise RuntimeError(
                        f"Pool audit: outgoing reuse exceeds held barriques at {node}."
                    )
                incoming_paths[destination].extend(selected)
                cursor += amount

        if next_path > model._n_barr:
            raise RuntimeError(
                f"Pool audit: solution requires {next_path} fresh barriques; "
                f"only {model._n_barr} exist."
            )
        for path, path_nodes in paths.items():
            for previous, following in zip(path_nodes, path_nodes[1:]):
                finish = pyo.value(model.Tf[previous])
                start = pyo.value(model.Ts[following])
                if finish > start + tol or start > finish + 24.0 + tol:
                    raise RuntimeError(
                        f"Pool audit: decomposed barrique path {path} is invalid."
                    )
    else:
        next_path = 0

    jar_intervals = []
    if hasattr(model, "NumJar"):
        for key in model.NumJar:
            count = integer_value(model.NumJar[key], f"NumJar{key}")
            if count:
                jar_intervals.append(
                    (key, pyo.value(model.Ts[key]), pyo.value(model.Tf[key]), count)
                )
        max_jar_use = 0
        for _, checkpoint, _, _ in jar_intervals:
            live = sum(
                count
                for _, start, finish, count in jar_intervals
                if start <= checkpoint + tol and finish > checkpoint + tol
            )
            max_jar_use = max(max_jar_use, live)
            if live > model._n_jar:
                raise RuntimeError(
                    f"Pool audit: jar occupancy {live} exceeds {model._n_jar} "
                    f"at time {checkpoint}."
                )
    else:
        max_jar_use = 0

    print(
        f"Pool audit passed: {next_path}/{model._n_barr} barrique paths; "
        f"maximum jar occupancy {max_jar_use}/{model._n_jar}."
    )
    return {"barrique_paths": next_path, "max_jar_occupancy": max_jar_use}


def solve_model(
    model,
    time_limit=3600 * 2,
    results_path="results_economic_gdp.txt",
    mipgap=0.03,
    tie_break_time_limit=60,
    branch_priorities=True,
    threads=6,
):
    solver = SolverFactory(solver_name)
    if solver is None:
        print(f"Error: Solver {solver_name} not found")
        return None

    # Set solver options
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = mipgap
    if threads is not None:
        solver.options["Threads"] = threads
    # solver.options["MIPFocus"] = 2
    # solver.options["Heuristics"] = 0.25
    # solver.options["Cuts"] = 2
    # solver.options["Symmetry"] = 2
    # solver.options["Presolve"] = 2
    # solver.options["Method"] = 2

    if solver_name == "gurobi_persistent":
        solver.set_instance(model)
        if branch_priorities:
            for (_, _), d_active in model._active_disjuncts.items():
                solver.set_var_attr(
                    d_active.binary_indicator_var, "BranchPriority", 100
                )
            for idx in model.y:
                solver.set_var_attr(model.y[idx], "BranchPriority", 50)
            print("Branching priorities enabled (task activation=100, assignment=50).")
        else:
            print("Branching priorities disabled.")
        print(f"Solving with {solver_name}...")
        results = solver.solve(tee=True)
    else:
        print(f"Solving with {solver_name}...")
        results = solver.solve(model, tee=True)

    # Lexicographic start-time tie-break
    primary_termination = results.solver.termination_condition
    reported_gap = results.solver.get("gap", None)
    exact_primary = (
        primary_termination == pyo.TerminationCondition.optimal
        and reported_gap is not None
        and float(reported_gap) <= 1e-9
    )
    if tie_break_time_limit and exact_primary:
        primary_value = pyo.value(model.PrimaryObjective)
        primary_snapshot = snapshot_variable_values(model)
        model.OBJ.deactivate()
        model.PrimaryObjectiveFloor = pyo.Constraint(
            expr=model.PrimaryObjective >= primary_value - 1e-6
        )
        model.TimePullOBJ = pyo.Objective(
            expr=sum(model.Ts[i, n] for i in model.i for n in model.n),
            sense=pyo.minimize,
        )
        if solver_name == "gurobi_persistent":
            solver.add_constraint(model.PrimaryObjectiveFloor)
            solver.set_objective(model.TimePullOBJ)
            solver.options["TimeLimit"] = tie_break_time_limit
            solver.options["MIPGap"] = 0.0
            tie_results = solver.solve(tee=True, warmstart=True)
        else:
            solver.options["TimeLimit"] = tie_break_time_limit
            solver.options["MIPGap"] = 0.0
            tie_results = solver.solve(model, tee=True, warmstart=True)
        tie_ok = tie_results.solver.termination_condition in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit,
        }
        retained_primary = pyo.value(model.PrimaryObjective, exception=False)
        if (
            not tie_ok
            or retained_primary is None
            or retained_primary < primary_value - 2e-6
        ):
            restore_variable_values(primary_snapshot)
            print(
                "Secondary time tie-break failed validation; restored primary solution."
            )
        else:
            print(
                "Applied lexicographic start-time tie-break without changing "
                f"the primary objective ({retained_primary:.6f})."
            )
        model.TimePullOBJ.deactivate()
        model.OBJ.activate()
    elif tie_break_time_limit:
        print(
            "Skipped start-time tie-break because the primary solve did not "
            "report a zero optimality gap."
        )

    termination = results.solver.termination_condition
    has_incumbent = _has_solver_incumbent(results)
    accepted_termination = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    }
    if has_incumbent:
        _audit_pool_solution(model)
        if termination == pyo.TerminationCondition.maxTimeLimit:
            print(
                "Time limit reached with a feasible incumbent; exporting the "
                f"incumbent (reported MIP gap: {results.solver.get('gap', 'unavailable')})."
            )
        elif termination not in accepted_termination:
            print(
                f"Solver stopped with termination condition {termination}; "
                "exporting the reported feasible incumbent."
            )
        print(f"Objective: {pyo.value(model.OBJ, exception=False)}")
        if hasattr(model, "Revenue"):
            print(
                f"Profit:              {pyo.value(model.Profit, exception=False):.2f}"
            )
            print(
                f"  Revenue:           {pyo.value(model.Revenue, exception=False):.2f}"
            )
            print(
                f"  Grape Skin Rev:    {pyo.value(model.GrapeSkinRevenue, exception=False):.2f}"
            )
            print(
                f"  Outsourcing Cost: -{pyo.value(model.OutsourcingCost, exception=False):.2f}"
            )
            print(
                f"  Raw Material Cost:-{pyo.value(model.RawMaterialCost, exception=False):.2f}"
            )
            print(
                f"  Lateness Cost:    -{pyo.value(model.LatenessCost, exception=False):.2f}"
            )
            print(
                f"  Discard Cost:     -{pyo.value(model.DiscardCost, exception=False):.2f}"
            )
            print(
                f"  Cooling Cost:     -{pyo.value(model.CoolingCost, exception=False):.2f}"
            )
            print("Scheduling penalties (not in Profit):")
            print(
                f"  Stg Holding Cost: -{pyo.value(model.StgHoldingCost, exception=False):.2f}"
            )
            print(
                f"  Makespan:         -{pyo.value(model.MakespanPenalty, exception=False):.2f}"
            )
            print(
                f"  Time Pull metric: {pyo.value(model.TimePullPenalty, exception=False):.2f}"
            )
        export_results(
            model,
            "Wine Scheduling Economic GDP",
            results_path,
        )
    else:
        print("No solution or Infeasible")

    return results


if __name__ == "__main__":
    import os
    import sys

    params_file = "parameters.toml"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--params":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                params_file = args[i + 1]
        elif a.startswith("--params="):
            params_file = a.split("=", 1)[1]

    stem = os.path.splitext(os.path.basename(params_file))[0]
    suffix = stem.replace("parameters", "", 1)
    results_path = f"results_economic_gdp{suffix}.txt"
    try:
        model = create_wine_scheduling_model(toml_file=params_file)
        solve_model(
            model,
            results_path=results_path,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
