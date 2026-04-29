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

from utils import export_results

# Apply Transformation, set to "hull" or "bigm"
transformation_type = "hull"


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


def create_wine_scheduling_model(toml_file="parameters.toml"):
    """Create and return the wine scheduling economic optimization model with GDP"""

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
        # Optional trailing Alm should not hide lateness when it is skipped
        if len(cfg["steps"]) >= 2 and cfg["steps"][-1] == "Alm":
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
    # alm_int_lines: lines where Alm is an intermediate buffer before an aging step
    alm_int_lines = {
        ln
        for ln, cfg in lines_cfg.items()
        if "Alm" in cfg["steps"] and cfg["steps"][-1] != "Alm"
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

    model.i = pyo.Set(initialize=tasks)
    model.iAlmInt = pyo.Set(initialize=[f"Alm{ln}" for ln in alm_int_lines])
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
            if "Alm" in unit_cfg.get("task_bounds", {}):
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
        prefix = task_stage(task)

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

    model.n = pyo.Set(initialize=list(range(1, params["global"]["n_max"] + 1)))

    # States - built dynamically from line step definitions:
    #   s_red / s_white_rose: shared raw-material pools by product category
    #   m: shared liquid must pool (all pressing lines produce into / consume from)
    #   v{l}: after Fa (zero-wait intermediate)
    #   vl{l}: after Fl (NIS intermediate, only when Fl is in steps)
    #   product state: final product key declared on each line
    #   dsch: discard / discharge balance state
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
        if line in alm_int_lines:
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
        elif stage == "Alm":
            if line in alm_int_lines:
                # Intermediate buffer: consume pre-aging state, produce vbuf (can wait)
                pre_ag = f"v{line}" if line in no_fl_lines else f"vl{line}"
                ICS_data.append((task, pre_ag))
                IPS_data.append((task, f"vbuf{line}"))
        elif is_aging_stage(stage):
            if line in aging_before_cs_lines:
                # Aging before Cs: consume buffer state if Alm exists, else fermentation intermediate
                if line in alm_int_lines:
                    ICS_data.append((task, f"vbuf{line}"))
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
    )
    model.SNIS = pyo.Set(
        initialize=[f"vl{ln}" for ln in lines if ln not in no_fl_lines]
        + [f"va{ln}" for ln in aging_before_cs_lines]
    )

    tc1_data = []
    for line in lines:
        steps = lines_cfg[line]["steps"]
        for idx in range(len(steps) - 1):
            current_task = f"{steps[idx]}{line}"
            next_task = f"{steps[idx + 1]}{line}"
            if steps[idx] == "Alm" and line in alm_int_lines:
                pass
            elif steps[idx] == "Pr" and steps[idx + 1] == "Fa":
                pass
            else:
                tc1_data.append((current_task, next_task))
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
        prefix = task_stage(task)

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

    # Intermediate Alm rho: pass-through buffer (consume pre-aging, produce vbuf)
    for line in alm_int_lines:
        task = f"Alm{line}"
        pre_ag = f"v{line}" if line in no_fl_lines else f"vl{line}"
        model.rhoIScons[task, pre_ag] = -1.0
        model.rhoISprod[task, f"vbuf{line}"] = 1.0

    # Override aging task rho for alm_int_lines: consume vbuf (not the pre-aging state)
    # TODO: We still need to update this
    for line in alm_int_lines:
        pre_ag = f"v{line}" if line in no_fl_lines else f"vl{line}"
        steps = lines_cfg[line]["steps"]
        for stage in steps:
            if is_aging_stage(stage):
                task = f"{stage}{line}"
                model.rhoIScons[task, pre_ag] = 0.0
                model.rhoIScons[task, f"vbuf{line}"] = -1.0

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

    for i in model.i:
        if i in barr_tasks:
            b_ub_i = barr_cap * n_barr
        elif i in jar_tasks:
            b_ub_i = jar_cap * n_jar
        else:
            stage_i = task_stage(i)
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
        for n in model.n:
            model.b[i, n].setub(b_ub_i)

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
    model.Discard = pyo.Var(model.SI, model.n, domain=pyo.NonNegativeReals)
    for s in model.SI:
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

    model.task_jmax = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.jMax)
    model.task_jmin = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.jMin)
    model.task_imax = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.iMax)
    pressing_lines_pr = [ln for ln in lines if "Pr" in lines_cfg[ln]["steps"]]
    if "Pr" in model.i and pressing_lines_pr:
        n_young_pressing = sum(1 for ln in pressing_lines_pr if ln in set(young_lines))
        n_aged_pressing = len(pressing_lines_pr) - n_young_pressing
        min_K_pr = min(len(lines_cfg[ln]["steps"]) for ln in pressing_lines_pr)
        n_valid_pr = n_max - (min_K_pr - 1)
        model.task_imax["Pr"] = min(
            n_young_pressing * imax_young + n_aged_pressing, n_valid_pr
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
        stage = task_stage(task)
        template = params.get("templates", {}).get(stage, {})
        if "max_units" in template:
            model.task_jmax[task] = int(template["max_units"])
        if "min_units" in template:
            model.task_jmin[task] = int(template["min_units"])

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

    # Global lower bound for makespan (longest line path)
    min_makespan = max(
        sum(
            pyo.value(model.alpha["Pr" if step == "Pr" else f"{step}{line}"])
            for step in cfg["steps"]
        )
        for line, cfg in lines_cfg.items()
    )
    model.ms_lb = pyo.Constraint(expr=model.MS >= min_makespan)

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
    # Per-task time window bounds derived from minimum cumulative step durations.
    # earliest_start[k] = sum of alpha for all steps before k (can't start earlier).
    # latest_finish[k] = H - sum of alpha for all steps after k (must leave room).
    H_val = pyo.value(model.H)
    task_ES: dict[str, float] = {}
    task_LF: dict[str, float] = {}
    for line, cfg in lines_cfg.items():
        steps = cfg["steps"]
        tnames = ["Pr" if s == "Pr" else f"{s}{line}" for s in steps]
        alphas = [pyo.value(model.alpha[t]) for t in tnames]
        prefix = [0.0] * len(steps)
        suffix = [0.0] * len(steps)
        for k in range(1, len(steps)):
            prefix[k] = prefix[k - 1] + alphas[k - 1]
        for k in range(len(steps) - 2, -1, -1):
            suffix[k] = suffix[k + 1] + alphas[k + 1]
        for k, (step, task) in enumerate(zip(steps, tnames)):
            lf = H_val - suffix[k]
            if step == "Pr":
                task_ES["Pr"] = 0.0
                if "Pr" not in task_LF or lf > task_LF["Pr"]:
                    task_LF["Pr"] = lf
                continue
            task_ES[task] = prefix[k]
            task_LF[task] = lf
            if prefix[k] > 0:
                for n in model.n:
                    model.Ts[task, n].setlb(prefix[k])
            if suffix[k] > 0:
                for n in model.n:
                    model.Tf[task, n].setub(lf)
            if lf < H_val:
                for n in model.n:
                    model.Ts[task, n].setub(lf)

    if "Pr" in task_LF:
        pr_lf = task_LF["Pr"]
        if pr_lf < H_val:
            for n in model.n:
                model.Tf["Pr", n].setub(pr_lf)
                model.Ts["Pr", n].setub(pr_lf)

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
    # STATIC FIXING: impossible task-event combinations
    # ========================================
    # Task at position k (1-indexed) in a line of K steps needs:
    #   min valid event = k
    #   max valid event = n_max-(K-k)
    task_pos_len: dict[str, tuple[int, int]] = {}
    pressing_lines_all = [ln for ln in lines if "Pr" in lines_cfg[ln]["steps"]]
    if pressing_lines_all:
        min_K_pr = min(len(lines_cfg[ln]["steps"]) for ln in pressing_lines_all)
        task_pos_len["Pr"] = (1, min_K_pr)
    for line, cfg in lines_cfg.items():
        steps = cfg["steps"]
        K = len(steps)
        for k, step in enumerate(steps, start=1):
            if step == "Pr":
                continue
            task_pos_len[f"{step}{line}"] = (k, K)

    n_fixed = 0
    task_valid_event_count: dict[str, int] = {}
    for i in model.i:
        k, K = task_pos_len[i]
        min_event = k
        max_event = n_max - (K - k)
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

    # Cross-event pool sequencing pairs: For each unordered pair {(i1,n1),(i2,n2)} of
    # distinct pool task-events that are not both fixed-inactive we add two
    # sequencing binaries (z_fwd / z_rev) and three constraints that correctly handle:
    #   - z_fwd = 1  ->  Tf[i1,n1] <= Ts[i2,n2]        (i1 finishes before i2 starts)
    #   - z_rev = 1  ->  Tf[i2,n2] <= Ts[i1,n1]        (i2 finishes before i1 starts)
    #   - z_fwd=z_rev=0  ->  concurrent, so combined count must <= pool size
    # BarriqueMaxIdle fires as a side-effect of z_fwd=1 (or z_rev=1)
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

    barr_cross = [(i1, n1, i2, n2) for (i1, n1), (i2, n2) in combinations(barr_te, 2)]
    jar_cross = [(i1, n1, i2, n2) for (i1, n1), (i2, n2) in combinations(jar_te, 2)]
    print(
        f"  [pool seq] Barrique cross-event pairs: {len(barr_cross)}, "
        f"Jar cross-event pairs: {len(jar_cross)}"
    )

    if barr_cross:
        model.BarrSeqFwd = pyo.Var(barr_cross, domain=pyo.Binary)
        model.BarrSeqRev = pyo.Var(barr_cross, domain=pyo.Binary)
    if jar_cross:
        model.JarSeqFwd = pyo.Var(jar_cross, domain=pyo.Binary)
        model.JarSeqRev = pyo.Var(jar_cross, domain=pyo.Binary)

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
        return sum(terms) <= model.H

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
        return model.ST[s, n_max] + produced - model.Discard[s, n_max] == 0

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
        <= n_barr * active_disjuncts[i, n].binary_indicator_var,
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
        <= n_jar * active_disjuncts[i, n].binary_indicator_var,
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

    # Pool capacity per event: all barrique/jar tasks compete for the same physical pool
    model.BarrCapacity = pyo.Constraint(
        model.n,
        rule=lambda m, n: sum(m.NumBarr[i, n] for i in m.iBAR) <= n_barr,
    )
    model.JarCapacity = pyo.Constraint(
        model.n,
        rule=lambda m, n: sum(m.NumJar[i, n] for i in m.iJAR) <= n_jar,
    )

    # A06: Max activations (non-storage tasks)
    model.A06 = pyo.Constraint(
        model.inst,
        rule=lambda m, i: sum(active_disjuncts[i, n].binary_indicator_var for n in m.n)
        <= m.task_imax[i],
    )

    # A06_min: Min activations valid cuts for non-outsourceable lines
    min_act_indices = []
    min_act_rhs: dict[str, int] = {}

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
            rhs = min(
                rhs,
                task_valid_event_count.get(task_name, rhs),
                int(pyo.value(model.task_imax[task_name])),
            )
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

    # Ecobulk Alm max duration: 720 h (1 month). Big-M relaxed when y[i,j,n]=0.
    # TODO: Update Alm
    all_alm = list(model.iAlmInt)
    eco_alm_index = [
        (i, j, n)
        for i in all_alm
        for j in model.JECO
        for n in model.n
        if (i, j) in model.ij_nonpool
    ]
    model.A_eco_alm_max = pyo.Constraint(
        eco_alm_index,
        rule=lambda m, i, j, n: m.Tf[i, n] - m.Ts[i, n]
        <= 720 + pyo.value(m.H) * (1 - m.y[i, j, n]),
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
    # product must go through that aging task.
    model.A18_aging_link = pyo.Constraint(
        [p for p in model.SMarket if p in model.aging_task_by_product],
        rule=lambda m, p: m.FinalProd[p]
        <= sum(m.b[m.aging_task_by_product[p], n] for n in m.n),
    )
    model.A21 = pyo.Constraint(
        model.ipst, [n_max], rule=lambda m, i, n: m.Tf[i, n] <= m.MS
    )

    # Products with an aging stage are not subject to the lateness penalty
    aging_products = set(aging_task_by_product.keys())

    # Ordering equality for N-way campaign disjuncts.
    # Single constraint: sum_{k=2}^{N} (k-1)*c_k = S (S = active prior event count)
    # encodes "campaign number = 1 + prior active events" for any N.
    model.CampOrderEq = pyo.ConstraintList()

    model.LateDefByProduct = pyo.ConstraintList()
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
                    model.CampOrderEq.add(
                        sum(
                            (camp_k - 1)
                            * camp_disjuncts_list[camp_k - 1].binary_indicator_var
                            for camp_k in range(2, max_camp + 1)
                        )
                        == S
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

            add_constr(
                d_active,
                "duration",
                model.Tf[i, n]
                == model.Ts[i, n] + model.alpha[i] + model.beta[i] * model.b[i, n],
            )

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

            add_constr(d_inactive, "b_zero", model.b[i, n] == 0)
            add_constr(d_inactive, "duration", model.Tf[i, n] == model.Ts[i, n])
            # Symmetry breaking: collapse idle event to previous event time.
            # Prevents Ts[i,n] floating freely in [0, H] when nothing happens.
            if n > 1:
                add_constr(
                    d_inactive, "ts_collapse", model.Ts[i, n] == model.Tf[i, n - 1]
                )

            # y_zero / bj_zero only needed for non-pool tasks
            if compatible_units:
                d_inactive.y_zero = pyo.ConstraintList()
                d_inactive.bj_zero = pyo.ConstraintList()
                for j in compatible_units:
                    d_inactive.y_zero.add(model.y[i, j, n] == 0)
                    d_inactive.bj_zero.add(model.bj[i, j, n] == 0)

            # Outer disjunction
            model.add_component(
                f"Disj_Task_{i}_{n}",
                gdp.Disjunction(expr=[d_active, d_inactive]),
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

    def M_bar_max(i_prev: str, i_next: str) -> float:
        """Big-M for BarriqueMaxIdle: Ts[next] <= Tf[prev]+24 + M*(1-IsHandoff)."""
        return max(0.0, task_LF[i_next] - task_ES[i_prev] - 24.0)

    def M_bar_ge(i_prev: str, i_next: str) -> float:
        """Big-M for handoff timing: Ts[next] >= Tf[prev] - M*(1-IsHandoff)."""
        return max(0.0, task_LF[i_prev] - task_ES[i_next])

    if barr_cross:
        model.BarrSeqFwdCon = pyo.Constraint(
            barr_cross,
            rule=lambda m, i1, n1, i2, n2: m.Tf[i1, n1]
            <= m.Ts[i2, n2] + M_bar_ge(i1, i2) * (1 - m.BarrSeqFwd[i1, n1, i2, n2]),
        )
        model.BarrSeqRevCon = pyo.Constraint(
            barr_cross,
            rule=lambda m, i1, n1, i2, n2: m.Tf[i2, n2]
            <= m.Ts[i1, n1] + M_bar_ge(i2, i1) * (1 - m.BarrSeqRev[i1, n1, i2, n2]),
        )
        model.BarrCapacityCross = pyo.Constraint(
            barr_cross,
            rule=lambda m, i1, n1, i2, n2: m.NumBarr[i1, n1] + m.NumBarr[i2, n2]
            <= n_barr
            + n_barr * (m.BarrSeqFwd[i1, n1, i2, n2] + m.BarrSeqRev[i1, n1, i2, n2]),
        )
        model.BarriqueMaxIdle = pyo.Constraint(
            barr_cross,
            rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
            <= m.Tf[i1, n1]
            + 24.0
            + M_bar_max(i1, i2) * (1 - m.BarrSeqFwd[i1, n1, i2, n2]),
        )
        model.BarriqueMaxIdleRev = pyo.Constraint(
            barr_cross,
            rule=lambda m, i1, n1, i2, n2: m.Ts[i1, n1]
            <= m.Tf[i2, n2]
            + 24.0
            + M_bar_max(i2, i1) * (1 - m.BarrSeqRev[i1, n1, i2, n2]),
        )

    if jar_cross:
        model.JarSeqFwdCon = pyo.Constraint(
            jar_cross,
            rule=lambda m, i1, n1, i2, n2: m.Tf[i1, n1]
            <= m.Ts[i2, n2] + M_bar_ge(i1, i2) * (1 - m.JarSeqFwd[i1, n1, i2, n2]),
        )
        model.JarSeqRevCon = pyo.Constraint(
            jar_cross,
            rule=lambda m, i1, n1, i2, n2: m.Tf[i2, n2]
            <= m.Ts[i1, n1] + M_bar_ge(i2, i1) * (1 - m.JarSeqRev[i1, n1, i2, n2]),
        )
        model.JarCapacityCross = pyo.Constraint(
            jar_cross,
            rule=lambda m, i1, n1, i2, n2: m.NumJar[i1, n1] + m.NumJar[i2, n2]
            <= n_jar
            + n_jar * (m.JarSeqFwd[i1, n1, i2, n2] + m.JarSeqRev[i1, n1, i2, n2]),
        )
        model.JarMaxIdle = pyo.Constraint(
            jar_cross,
            rule=lambda m, i1, n1, i2, n2: m.Ts[i2, n2]
            <= m.Tf[i1, n1]
            + 24.0
            + M_bar_max(i1, i2) * (1 - m.JarSeqFwd[i1, n1, i2, n2]),
        )
        model.JarMaxIdleRev = pyo.Constraint(
            jar_cross,
            rule=lambda m, i1, n1, i2, n2: m.Ts[i1, n1]
            <= m.Tf[i2, n2]
            + 24.0
            + M_bar_max(i2, i1) * (1 - m.JarSeqRev[i1, n1, i2, n2]),
        )

    # ========================================
    # PRECEDENCE (Conditional Constraints, Eq. 2.2)
    # ========================================

    # General: W_{i',n} and W_{i,n+1} both active implies Ts_{i,n+1} >= Tf_{i',n}
    model.prec_ge = pyo.Constraint(
        [
            (i_cons, i_prod, s, n)
            for i_cons, i_prod, s in precedence_pairs
            for n in model.n
            if n < n_max
        ],
        rule=lambda m, i_cons, i_prod, s, n: m.Ts[i_cons, n + 1]
        >= m.Tf[i_prod, n]
        - M_ge(i_prod, i_cons)
        * (
            2
            - active_disjuncts[i_prod, n].binary_indicator_var
            - active_disjuncts[i_cons, n + 1].binary_indicator_var
        ),
    )

    # ZW + NIS + Ecobulk: W_{i',n} and W_{i,n+1} both active implies Ts_{i,n+1} = Tf_{i',n}
    zw_nis_eco_pairs = [
        (i_cons, i_prod, s)
        for i_cons, i_prod, s in precedence_pairs
        if s in model.SZW or s in model.SNIS or s in model.SFISEco
    ]
    model.prec_le = pyo.Constraint(
        [
            (i_cons, i_prod, s, n)
            for i_cons, i_prod, s in zw_nis_eco_pairs
            for n in model.n
            if n < n_max
        ],
        rule=lambda m, i_cons, i_prod, s, n: m.Ts[i_cons, n + 1]
        <= m.Tf[i_prod, n]
        + M_le(i_prod, i_cons)
        * (
            2
            - active_disjuncts[i_prod, n].binary_indicator_var
            - active_disjuncts[i_cons, n + 1].binary_indicator_var
        ),
    )

    # ========================================
    # STORAGE PERSISTENCE (Logical Implications, Eq. 2.3)
    # ========================================

    # TODO: Update storage and storage persistance

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

    # Exact linearization of alpha[i] * bj[i,j,n] * W[i,n] for non-storage tasks.
    # W is binary, so the McCormick envelope is exact at integer points.
    model.alpha_cooling_index = pyo.Set(
        dimen=3,
        initialize=[
            (i, j, n)
            for i in model.inst
            for j in model.j
            for n in model.n
            if (i, j) in model.ij_nonpool
        ],
    )
    model.zAlphaCooling = pyo.Var(
        model.alpha_cooling_index, domain=pyo.NonNegativeReals
    )

    model.AlphaCoolingLin_lb = pyo.Constraint(
        model.alpha_cooling_index,
        rule=lambda m, i, j, n: m.zAlphaCooling[i, j, n]
        >= m.bj[i, j, n]
        - m.Bmax[i, j] * (1 - active_disjuncts[i, n].binary_indicator_var),
    )
    model.AlphaCoolingLin_ub_bj = pyo.Constraint(
        model.alpha_cooling_index,
        rule=lambda m, i, j, n: m.zAlphaCooling[i, j, n] <= m.bj[i, j, n],
    )
    model.AlphaCoolingLin_ub_w = pyo.Constraint(
        model.alpha_cooling_index,
        rule=lambda m, i, j, n: m.zAlphaCooling[i, j, n]
        <= m.Bmax[i, j] * active_disjuncts[i, n].binary_indicator_var,
    )

    # beta=0 is validated above, so cooling time = alpha[i] * W[i,n] only.
    # zAlphaCooling linearizes bj[i,j,n] * W[i,n] exactly (W is binary).
    def cooling_term(i, j, n):
        if i in model.inst:
            return model.alpha[i] * model.zAlphaCooling[i, j, n]
        return 0.0

    model.CoolingCost = pyo.Expression(
        expr=model.costCooling
        * sum(
            cooling_term(i, j, n)
            for (i, j) in model.ij_nonpool
            if j in model.JEXT
            for n in model.n
        )
    )

    model.DiscardCost = pyo.Expression(
        expr=model.penaltyDiscard
        * sum(model.Discard[s, n] for s in model.SI for n in model.n)
    )

    model.MakespanPenalty = pyo.Expression(expr=model.penaltyMS * model.MS)
    # epsilon-trick: tiny penalty pulling all task start times as early as possible.
    # Breaks the degeneracy of "floating" tasks that can shift freely without
    # changing the objective.
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
    )

    def obj_func(model):
        return (
            # model.Revenue
            +model.GrapeSkinRevenue
            - model.OutsourcingCost
            - model.LatenessCost
            - model.RawMaterialCost
            - model.MakespanPenalty
            - model.CoolingCost
            - model.DiscardCost
            - model.TimePullPenalty
        )

    model.OBJ = pyo.Objective(rule=obj_func, sense=pyo.maximize)

    # ========================================
    # LP TIGHTENING
    # ========================================
    # The helper must be called after y / Bmin / Bmax are fully
    # initialised and before the GDP transformation flattens the disjuncts.
    add_cover_cuts(model)

    print(f"Applying GDP {transformation_type} transformation...")
    pyo.TransformationFactory(f"gdp.{transformation_type}").apply_to(model)

    return model


def solve_model(model, solver_name="gurobi", time_limit=3600 * 2):
    solver = SolverFactory(solver_name)
    if solver is None:
        print(f"Error: Solver {solver_name} not found")
        return None

    # Set solver options
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = 0.001
    solver.options["MIPFocus"] = 2
    solver.options["Heuristics"] = 0.2
    solver.options["Cuts"] = 2
    solver.options["Symmetry"] = 2
    solver.options["Presolve"] = 2

    print(f"Solving with {solver_name}...")
    results = solver.solve(model, tee=True)

    if (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
        or results.solver.termination_condition == pyo.TerminationCondition.feasible
    ):
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
            print("Penalties:")
            print(
                f"  Makespan:         -{pyo.value(model.MakespanPenalty, exception=False):.2f}"
            )
            print(
                f"  Cooling:          -{pyo.value(model.CoolingCost, exception=False):.2f}"
            )
            print(
                f"  Time Pull:        -{pyo.value(model.TimePullPenalty, exception=False):.2f}"
            )
        export_results(
            model,
            "Wine Scheduling Economic GDP",
            "results_economic_gdp.txt",
        )
    else:
        print("No solution or Infeasible")

    return results


if __name__ == "__main__":
    try:
        model = create_wine_scheduling_model()
        solve_model(model)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
