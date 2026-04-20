"""
Wine Scheduling Economic Optimization Model using Pyomo GDP.

This script keeps the GDP scheduling formulation and switches the objective
to economic profit maximization using price, outsourcing, and lateness terms.
"""

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
        pairs = [
            (j, round(pyo.value(model.Bmax[i, j]), 6))
            for j in model.j
            if (i, j) in model.ij
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
    no_fl_lines = {l for l, cfg in lines_cfg.items() if "Fl" not in cfg["steps"]}
    ist_lines = {l for l, cfg in lines_cfg.items() if "Alm" in cfg["steps"]}
    eco_lines = {l for l, cfg in lines_cfg.items() if cfg.get("ecobulk", False)}
    # Lines where an aging step (AgeBar*, AgeJar*) precedes Cs
    aging_before_cs_lines = {
        l
        for l, cfg in lines_cfg.items()
        if "Cs" in cfg["steps"]
        and any(
            is_aging_stage(step) for step in cfg["steps"][: cfg["steps"].index("Cs")]
        )
    }

    model.i = pyo.Set(initialize=tasks)
    model.ist = pyo.Set(initialize=[f"Alm{l}" for l in ist_lines])
    model.inst = pyo.Set(initialize=[t for t in tasks if t not in model.ist])
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

    model.j = pyo.Set(initialize=unit_instances)
    model.JST = pyo.Set(initialize=storage_units)
    model.JEXT = pyo.Set(initialize=exterior_units)
    model.JBAR = pyo.Set(initialize=unit_instances_by_base.get("barrique", []))
    model.JJAR = pyo.Set(initialize=unit_instances_by_base.get("jar", []))

    barrique_units_ordered = sorted(model.JBAR, key=unit_order_key)
    jar_units_ordered = sorted(model.JJAR, key=unit_order_key)

    model.nJST = pyo.Param(initialize=len(model.JST))
    model.inv_nJST = pyo.Param(initialize=1 / len(model.JST) if len(model.JST) > 0 else 0.0)

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

    model.n = pyo.Set(initialize=list(range(1, params["global"]["n_max"] + 1)))

    # States - built dynamically from line step definitions:
    #   s_red / s_white_rose: shared raw-material pools by product category
    #   m{l}: liquid must after Pressing (only for pressed wines)
    #   v{l}: after Fa (zero-wait intermediate)
    #   vl{l}: after Fl (NIS intermediate, only when Fl is in steps)
    #   product state: final product key declared on each line
    #   dsch: discard / discharge balance state
    line_raw_state = {}
    raw_states = set()
    states = ["dsch"]
    for line, cfg in lines_cfg.items():
        product_key = line_product[line]
        category = params["products"][product_key].get("category", "").strip().lower()
        raw_state = "s_red" if category == "red" else "s_white_rose"
        line_raw_state[line] = raw_state
        raw_states.add(raw_state)
        if raw_state not in states:
            states.append(raw_state)
        if "Pr" in cfg["steps"]:
            states.append(f"m{line}")
        states.append(f"v{line}")
        if "Fl" in cfg["steps"]:
            states.append(f"vl{line}")
        if line in aging_before_cs_lines:
            states.append(f"va{line}")
        states.append(line_product[line])
    model.s = pyo.Set(initialize=states)

    # States logic
    IPS_data = []
    ICS_data = []
    for task in tasks:
        line = task_line(task)
        stage = task_stage(task)
        if stage == "Pr":
            ICS_data.append((task, line_raw_state[line]))
            IPS_data.append((task, f"m{line}"))
            IPS_data.append((task, "dsch"))
        elif stage == "Fa":
            if "Pr" in lines_cfg[line]["steps"]:
                ICS_data.append((task, f"m{line}"))
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
            ICS_data.append((task, line_product[line]))
            IPS_data.append((task, line_product[line]))
        elif is_aging_stage(stage):
            if line in aging_before_cs_lines:
                # Aging before Cs, consume fermentation intermediate, produce va{line}
                if line in no_fl_lines:
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
    model.SP = pyo.Set(initialize=["dsch"] + [line_product[l] for l in lines])
    model.SI = model.s - model.SP - model.SR
    model.SFISEco = pyo.Set(initialize=[line_product[l] for l in eco_lines])
    model.SFISBar = pyo.Set(
        initialize=[
            line_product[l] for l in lines if l not in eco_lines and l not in ist_lines
        ]
    )
    model.SZW = pyo.Set(
        initialize=[f"v{l}" for l in lines]
        + [f"m{l}" for l in lines if "Pr" in lines_cfg[l]["steps"]]
    )
    model.SNIS = pyo.Set(
        initialize=[f"vl{l}" for l in lines if l not in no_fl_lines]
        + [f"va{l}" for l in aging_before_cs_lines]
    )

    tc1_data = []
    tc2_data = []
    for line in lines:
        steps = lines_cfg[line]["steps"]
        for idx in range(len(steps) - 1):
            current_task = f"{steps[idx]}{line}"
            next_task = f"{steps[idx + 1]}{line}"
            if steps[idx + 1] == "Alm":
                tc2_data.append((current_task, next_task))
            else:
                tc1_data.append((current_task, next_task))
    model.tc1 = pyo.Set(initialize=tc1_data, dimen=2)
    model.tc2 = pyo.Set(initialize=tc2_data, dimen=2)
    model.prd = pyo.Set(initialize=[line_product[l] for l in lines])

    # ========================================
    # PARAMETERS
    # ========================================
    global_cfg = params.get("global", {})
    model.iMax = pyo.Param(initialize=global_cfg.get("iMax", 1))
    model.jMax = pyo.Param(initialize=global_cfg.get("jMax", 1))
    model.jMin = pyo.Param(initialize=global_cfg.get("jMin", 1))
    model.iMaxST = pyo.Param(initialize=global_cfg.get("iMaxST", 1))
    model.jMaxST = pyo.Param(initialize=global_cfg.get("jMaxST", 2))
    model.jMinST = pyo.Param(initialize=global_cfg.get("jMinST", 1))
    model.MustUseEcobulk = pyo.Param(initialize=global_cfg.get("MustUseEcobulk", 0))

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
                    for unit_inst in unit_instances_by_base[unit]:
                        if (task, unit_inst) in model.ij:
                            model.Bmin[task, unit_inst] = limits[0]
                            model.Bmax[task, unit_inst] = limits[1]

    total = 0.0
    for j in model.JST:
        tank_count = 0
        tank_range_sum = 0.0
        for i in model.ist:
            if (i, j) in model.ij:
                tank_range_sum += pyo.value(model.Bmax[i, j]) - pyo.value(
                    model.Bmin[i, j]
                )
                tank_count += 1
        if tank_count > 0:
            total += tank_range_sum / tank_count
    total_avg_range_value = total if total > 0.0 else 1.0

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

    demand_data = {k: v["demand"] for k, v in params["products"].items()}
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
    model.Deadline = pyo.Param(
        initialize=global_cfg.get("deadline", pyo.value(model.H))
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

    model.penaltyEmptyTank = pyo.Param(initialize=params["global"]["penaltyEmptyTank"])
    model.penaltyAir = pyo.Param(initialize=params["global"]["penaltyAir"])
    model.penaltyMaxUtilization = pyo.Param(
        initialize=params["global"]["penaltyMaxUtilization"]
    )
    model.penaltyMS = pyo.Param(initialize=params["global"].get("penaltyMS", 0.0))
    model.costCooling = pyo.Param(initialize=params["global"].get("costCooling", 0.0))
    model.total_avg_range = pyo.Param(initialize=total_avg_range_value, mutable=False)
    model.inv_total_avg_range = pyo.Param(initialize=1 / total_avg_range_value)

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

    model.y = pyo.Var(model.ij, model.n, domain=pyo.Binary)

    # Continuous Variables
    model.b = pyo.Var(model.i, model.n, domain=pyo.NonNegativeReals)
    model.bj = pyo.Var(model.ij, model.n, domain=pyo.NonNegativeReals)

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

    for i in model.i:
        b_ub_i = sum(pyo.value(model.Bmax[i, j]) for j in model.j if (i, j) in model.ij)
        if b_ub_i <= 0:
            b_ub_i = max_tank_cap * len(model.j)  # Fallback for unexpected sparse data
        for n in model.n:
            model.b[i, n].setub(b_ub_i)

    for i, j in model.ij:
        for n in model.n:
            model.bj[i, j, n].setub(pyo.value(model.Bmax[i, j]))

    model.ST = pyo.Var(model.s, model.n, domain=pyo.NonNegativeReals)
    model.Ts = pyo.Var(
        model.i, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    model.Tf = pyo.Var(
        model.i, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    model.Tsj = pyo.Var(
        model.j, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    model.Tfj = pyo.Var(
        model.j, model.n, domain=pyo.NonNegativeReals, bounds=(0, model.H)
    )
    model.FinalProd = pyo.Var(model.SP, domain=pyo.NonNegativeReals)
    model.Outsource = pyo.Var(model.SMarket, domain=pyo.NonNegativeReals)
    model.MS = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, model.H))
    model.LatenessProd = pyo.Var(model.SMarket, domain=pyo.NonNegativeReals)
    model.JST_unused = pyo.Var(domain=pyo.NonNegativeReals)
    model.Freespace = pyo.Var(model.j, domain=pyo.NonNegativeReals)

    model.task_jmax = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.jMax)
    model.task_jmin = pyo.Param(model.i, mutable=True, initialize=lambda m, i: m.jMin)
    for task in model.i:
        stage = task_stage(task)
        template = params.get("templates", {}).get(stage, {})
        if "max_units" in template:
            model.task_jmax[task] = int(template["max_units"])
        if "min_units" in template:
            model.task_jmin[task] = int(template["min_units"])

    # Global lower bound for makespan (longest line path)
    min_makespan = max(
        sum(
            pyo.value(model.alpha[f"{step}{line}"])
            for step in cfg["steps"]
            if f"{step}{line}" in model.i
        )
        for line, cfg in lines_cfg.items()
    )
    model.ms_lb = pyo.Constraint(expr=model.MS >= min_makespan)

    FinalProd_bounds = params["product_ub"]
    for sp in model.SP:
        if sp in FinalProd_bounds:
            model.FinalProd[sp].setub(FinalProd_bounds[sp])

    late_ub = pyo.value(model.H) - pyo.value(model.Deadline)
    for s in model.SMarket:
        model.LatenessProd[s].setub(late_ub)
        model.Outsource[s].setub(pyo.value(model.D[s]))
    model.JST_unused.setub(pyo.value(model.nJST))
    for j in model.j:
        freespace_ub = max(
            pyo.value(model.Bmax[i, j]) for i in model.i if (i, j) in model.ij
        )
        model.Freespace[j].setub(freespace_ub)

    # Per-task time window bounds derived from minimum cumulative step durations.
    # earliest_start[k] = sum of alpha for all steps before k (can't start earlier).
    # latest_finish[k] = H - sum of alpha for all steps after k (must leave room).
    H_val = pyo.value(model.H)
    for line, cfg in lines_cfg.items():
        steps = cfg["steps"]
        alphas = [pyo.value(model.alpha[f"{step}{line}"]) for step in steps]
        prefix = [0.0] * len(steps)
        suffix = [0.0] * len(steps)
        for k in range(1, len(steps)):
            prefix[k] = prefix[k - 1] + alphas[k - 1]
        for k in range(len(steps) - 2, -1, -1):
            suffix[k] = suffix[k + 1] + alphas[k + 1]
        for k, step in enumerate(steps):
            task = f"{step}{line}"
            if prefix[k] > 0:
                for n in model.n:
                    model.Ts[task, n].setlb(prefix[k])
            if suffix[k] > 0:
                for n in model.n:
                    model.Tf[task, n].setub(H_val - suffix[k])

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
    # CONSTRAINTS (Standard Algebraic)
    # ========================================

    # Material balances (h04, h03)
    def h04_rule(model, s, n):
        consumed = sum(
            model.rhoIScons[i, s] * model.b[i, n]
            for i in model.i
            if (i, s) in model.ICS
        )
        return model.ST[s, n] == model.ST0[s] + consumed

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
        return model.ST[s, n] == model.ST[s, n - 1] + produced + consumed

    model.h03 = pyo.Constraint(model.s, [n for n in model.n if n > 1], rule=h03_rule)

    # Task sequencing g06 (Always active)
    def g06_rule(model, i, n):
        return model.Ts[i, n + 1] >= model.Tf[i, n]

    model.g06 = pyo.Constraint(
        model.i, [n for n in model.n if n < n_max], rule=g06_rule
    )

    # Unit capacity h09
    def h09_rule(model, j):
        total_time = sum(
            model.alpha[i] * model.y[i, j, n] + model.beta[i] * model.bj[i, j, n]
            for i in model.inst
            for n in model.n
            if (i, j) in model.ij
        )
        return total_time <= model.H

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

    model.h16 = pyo.Constraint(model.SZW, rule=h16_rule)

    model.g17 = pyo.Constraint(
        model.SMarket, rule=lambda m, s: m.FinalProd[s] + m.Outsource[s] >= m.D[s]
    )
    model.g17_no_outsource = pyo.Constraint(
        [s for s in model.SMarket if pyo.value(model.OutsourceAllowed[s]) == 0],
        rule=lambda m, s: m.Outsource[s] == 0,
    )

    # A01/A02/A01st/A02st: unit-count bounds using disjunct binary_indicator_var as W
    def A01_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            <= model.task_jmax[i] * active_disjuncts[i, n].binary_indicator_var
        )

    model.A01 = pyo.Constraint(model.inst, model.n, rule=A01_rule)

    def A02_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            >= model.task_jmin[i] * active_disjuncts[i, n].binary_indicator_var
        )

    model.A02 = pyo.Constraint(model.inst, model.n, rule=A02_rule)

    def A01st_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            <= model.jMaxST * active_disjuncts[i, n].binary_indicator_var
        )

    model.A01st = pyo.Constraint(model.ist, model.n, rule=A01st_rule)

    def A02st_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            >= model.jMinST * active_disjuncts[i, n].binary_indicator_var
        )

    model.A02st = pyo.Constraint(model.ist, model.n, rule=A02st_rule)

    # A04: Total batch
    def A04_rule(model, i, n):
        return model.b[i, n] == sum(
            model.bj[i, j, n] for j in model.j if (i, j) in model.ij
        )

    model.A04 = pyo.Constraint(model.i, model.n, rule=A04_rule)

    # A06: Max activations (non-storage tasks)
    model.A06 = pyo.Constraint(
        model.inst,
        rule=lambda m, i: sum(active_disjuncts[i, n].binary_indicator_var for n in m.n)
        <= m.iMax,
    )
    # A06st: Max activations (storage tasks)
    # With persistence (Eq. 2.3), W[i,n] is non-decreasing, so W[i, n_max] suffices.
    model.A06st = pyo.Constraint(
        model.ist,
        rule=lambda m, i: active_disjuncts[i, n_max].binary_indicator_var <= m.iMaxST,
    )
    model.A07 = pyo.Constraint(
        model.ist, model.n, rule=lambda m, i, n: m.Tf[i, n] >= m.Ts[i, n]
    )

    # A09/A10: Unit event sequencing
    model.A09 = pyo.Constraint(
        model.j,
        [n for n in model.n if n < n_max],
        rule=lambda m, j, n: m.Tsj[j, n + 1] >= m.Tfj[j, n],
    )
    model.A10 = pyo.Constraint(
        model.j, model.n, rule=lambda m, j, n: m.Tfj[j, n] >= m.Tsj[j, n]
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

    def A13b_rule(model, i, ip, n):
        return (
            active_disjuncts[ip, n + 1].binary_indicator_var
            == active_disjuncts[i, n].binary_indicator_var
        )

    model.A13b = pyo.Constraint(
        [
            (i, ip, n)
            for i, ip in model.tc2
            for n in model.n
            if n < n_max and model.MustUseEcobulk == 1
        ],
        rule=A13b_rule,
    )

    def A13c_rule(model, i, ip, n):
        return (
            active_disjuncts[ip, n + 1].binary_indicator_var
            <= active_disjuncts[i, n].binary_indicator_var
        )

    model.A13c = pyo.Constraint(
        [
            (i, ip, n)
            for i, ip in model.tc2
            for n in model.n
            if n < n_max and model.MustUseEcobulk != 1
        ],
        rule=A13c_rule,
    )

    # Storage unit persistence: y_{i,j,n} implies y_{i,j,n+1} (linear form, Eq. 2.3)
    model.A15 = pyo.Constraint(
        [
            (i, j, n)
            for i in model.ist
            for j in model.j
            for n in model.n
            if (i, j) in model.ij and n < n_max
        ],
        rule=lambda m, i, j, n: m.y[i, j, n + 1] >= m.y[i, j, n],
    )
    model.A17 = pyo.Constraint(
        [s for s in model.SFISEco if model.MustUseEcobulk == 1],
        rule=lambda m, s: sum(
            active_disjuncts[i, n].binary_indicator_var
            for i in m.i
            for n in m.n
            if (i, s) in m.ICS
        )
        >= 1,
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
    model.A22 = pyo.Constraint(
        rule=lambda m: m.JST_unused
        == m.nJST
        - sum(m.y[i, j, n_max] for j in m.JST for i in m.ist if (i, j) in m.ij)
    )

    def A23_rule(model, j, i):
        used = sum(model.bj[i, j, n] for n in model.n)
        used_flag = sum(model.y[i, j, n] for n in model.n)
        return model.Freespace[j] <= model.Bmax[i, j] - used + model.Bmax[i, j] * (
            1 - used_flag
        )

    model.A23 = pyo.Constraint(
        model.JST,
        model.ist,
        rule=lambda m, j, i: A23_rule(m, j, i)
        if (i, j) in m.ij
        else pyo.Constraint.Skip,
    )

    def A24_rule(model, j, i):
        used = sum(model.bj[i, j, n] for n in model.n)
        used_flag = sum(model.y[i, j, n] for n in model.n)
        return model.Freespace[j] >= model.Bmax[i, j] - used - model.Bmax[i, j] * (
            1 - used_flag
        )

    model.A24 = pyo.Constraint(
        model.JST,
        model.ist,
        rule=lambda m, j, i: A24_rule(m, j, i)
        if (i, j) in m.ij
        else pyo.Constraint.Skip,
    )

    # Products with an aging stage are not subject to the lateness penalty
    aging_products = set(aging_task_by_product.keys())

    model.LateDefByProduct = pyo.ConstraintList()
    for line in lines:
        product = line_product[line]
        if product in aging_products:
            continue
        i_last = line_lateness_task[line]
        # Tightest Big-M is achieved when the task is inactive Tf <= H is always
        # satisfied, so M only needs to cover H - Deadline
        M_late = pyo.value(model.H) - pyo.value(model.Deadline)
        for n in model.n:
            model.LateDefByProduct.add(
                model.Tf[i_last, n]
                <= model.Deadline
                + model.LatenessProd[product]
                + M_late * (1 - active_disjuncts[i_last, n].binary_indicator_var)
            )

    # ========================================
    # DISJUNCTIONS (Combined Task-Unit, Eq. 2.1)
    # ========================================

    def add_constr(block, name, expr):
        block.add_component(name, pyo.Constraint(expr=expr))

    # assign_disjuncts stores unit-selection sub-disjuncts (used in LogicalConstraints)
    assign_disjuncts = {}

    for i in model.i:
        compatible_units = [j for j in model.j if (i, j) in model.ij]
        for n in model.n:
            # Retrieve pre-created disjuncts, binary_indicator_var IS W_{i,n}
            d_active = active_disjuncts[i, n]

            # Duration equation for non-storage tasks only.
            # Storage persistence is enforced through logical assignment/activity
            # persistence across events, not by forcing Tf = H.
            if i not in model.ist:
                add_constr(
                    d_active,
                    "duration",
                    model.Tf[i, n]
                    == model.Ts[i, n] + model.alpha[i] + model.beta[i] * model.b[i, n],
                )

            # Terminal storage event: if active at n_max, enforce Tf == MS locally.
            if i in model.ist and n == n_max:
                add_constr(d_active, "horizon_sync_lb", model.Tf[i, n_max] >= model.MS)
                add_constr(d_active, "horizon_sync_ub", model.Tf[i, n_max] <= model.MS)

            # Nested unit selection: ⋁_{j in J_i}
            for j in compatible_units:
                d_assign = gdp.Disjunct()
                d_skip = gdp.Disjunct()
                d_active.add_component(f"d_assign_{j}", d_assign)
                d_active.add_component(f"d_skip_{j}", d_skip)
                assign_disjuncts[i, j, n] = d_assign

                # Assign: y=1, batch limits, time sync
                add_constr(d_assign, "y_on", model.y[i, j, n] == 1)
                add_constr(d_assign, "bmin", model.bj[i, j, n] >= model.Bmin[i, j])
                add_constr(d_assign, "bmax", model.bj[i, j, n] <= model.Bmax[i, j])
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

    # A03: At most one task per unit per event
    model.A03 = pyo.Constraint(
        model.j,
        model.n,
        rule=lambda m, j, n: sum(m.y[i, j, n] for i in m.i if (i, j) in m.ij) <= 1,
    )

    # Symmetry-breaking for indexed vessel pools: use lower index before higher.
    barrique_pairs = list(zip(barrique_units_ordered[:-1], barrique_units_ordered[1:]))
    if barrique_pairs:
        # Cumulative (cross-event) symmetry-breaking: barrique j_next cannot be
        # first used before barrique j_prev across the whole horizon.
        model.BarriqueSymmetryBreak = pyo.Constraint(
            barrique_pairs,
            model.n,
            rule=lambda m, j_prev, j_next, n: sum(
                m.y[i, j_next, n_prime]
                for i in m.i
                for n_prime in m.n
                if (i, j_next) in m.ij and n_prime <= n
            )
            <= sum(
                m.y[i, j_prev, n_prime]
                for i in m.i
                for n_prime in m.n
                if (i, j_prev) in m.ij and n_prime <= n
            ),
        )

    jar_pairs = list(zip(jar_units_ordered[:-1], jar_units_ordered[1:]))
    if jar_pairs:
        model.JarSymmetryBreak = pyo.Constraint(
            jar_pairs,
            model.n,
            rule=lambda m, j_prev, j_next, n: sum(
                m.y[i, j_next, n_prime]
                for i in m.i
                for n_prime in m.n
                if (i, j_next) in m.ij and n_prime <= n
            )
            <= sum(
                m.y[i, j_prev, n_prime]
                for i in m.i
                for n_prime in m.n
                if (i, j_prev) in m.ij and n_prime <= n
            ),
        )

    # Barrique reuse constraints: if a barrique is reused at a later event,
    # enforce a 1h minimum wait (cleaning) and a 24h maximum idle time
    # (barrique cannot sit empty for more than 24h or it deteriorates).
    barrique_reuse_index = [
        (j, i_prev, n_prev, i_next, n_next)
        for j in model.JBAR
        for i_prev in model.i
        for i_next in model.i
        for n_prev in model.n
        for n_next in model.n
        if (i_prev, j) in model.ij and (i_next, j) in model.ij and n_prev < n_next
    ]
    model.BarriqueMinWait = pyo.Constraint(
        barrique_reuse_index,
        rule=lambda m, j, i_prev, n_prev, i_next, n_next: m.Ts[i_next, n_next]
        >= m.Tf[i_prev, n_prev]
        + 1.0
        - m.H * (2 - m.y[i_prev, j, n_prev] - m.y[i_next, j, n_next]),
    )
    model.BarriqueMaxIdle = pyo.Constraint(
        barrique_reuse_index,
        rule=lambda m, j, i_prev, n_prev, i_next, n_next: m.Ts[i_next, n_next]
        <= m.Tf[i_prev, n_prev]
        + 24.0
        + m.H * (2 - m.y[i_prev, j, n_prev] - m.y[i_next, j, n_next]),
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
        - m.H
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
        + m.H
        * (
            2
            - active_disjuncts[i_prod, n].binary_indicator_var
            - active_disjuncts[i_cons, n + 1].binary_indicator_var
        ),
    )

    # ========================================
    # STORAGE PERSISTENCE (Logical Implications, Eq. 2.3)
    # ========================================

    # Y_{i,n} implies Y_{i,n+1}  for all i in I^st, n < N
    for i in model.ist:
        for n in model.n:
            if n < n_max:
                model.add_component(
                    f"persist_task_{i}_{n}",
                    pyo.LogicalConstraint(
                        expr=active_disjuncts[i, n].indicator_var.implies(
                            active_disjuncts[i, n + 1].indicator_var
                        )
                    ),
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
    late_products = [s for s in model.SMarket if s not in aging_task_by_product]  # type: ignore[union-attr]
    model.Lateness = pyo.Expression(
        expr=sum(model.LatenessProd[s] for s in late_products)
    )
    model.LatenessCost = pyo.Expression(
        expr=model.penaltyLate * sum(model.LatenessProd[s] for s in late_products)  # type: ignore[operator]
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

    # Exact linearization of alpha[i] * bj[i,j,n] * W[i,n] for non-storage tasks.
    # W is binary, so the McCormick envelope is exact at integer points.
    model.alpha_cooling_index = pyo.Set(
        dimen=3,
        initialize=[
            (i, j, n)
            for i in model.inst
            for j in model.j
            for n in model.n
            if (i, j) in model.ij
        ],
    )
    model.zAlphaCooling = pyo.Var(model.alpha_cooling_index, domain=pyo.NonNegativeReals)

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

    # For non-storage tasks the hull transformation enforces globally:
    #   Tf[i,n] - Ts[i,n] = alpha[i]*W[i,n] + beta[i]*b[i,n]
    # Substituting eliminates Tf and Ts from the bilinear product, replacing
    # variables with range [0, H] with W in [0,1] and b in [0,b_ub].
    def cooling_term(i, j, n):
        bj = model.bj[i, j, n]
        if i in model.inst:
            return (
                model.alpha[i] * model.zAlphaCooling[i, j, n]
                + model.beta[i] * bj * model.b[i, n]
            )
        # Exclude Alm storage cooling from the objective to keep a MILP model.
        return 0.0

    model.CoolingCost = pyo.Expression(
        expr=model.costCooling
        * sum(
            cooling_term(i, j, n)
            for (i, j) in model.ij
            if j in model.JEXT
            for n in model.n
        )
    )

    def obj_func(model):
        penalty_unused = model.penaltyEmptyTank * model.inv_nJST * model.JST_unused
        penalty_space = (
            model.penaltyAir
            * model.inv_total_avg_range
            * sum(model.Freespace[j] for j in model.JST)
        )
        return (
            model.Revenue
            - model.OutsourcingCost
            - model.LatenessCost
            - model.RawMaterialCost
            - penalty_unused
            - penalty_space
            - model.penaltyMS * model.MS
            - model.CoolingCost
        )

    model.OBJ = pyo.Objective(rule=obj_func, sense=pyo.maximize)

    # ========================================
    # LP TIGHTENING
    # ========================================
    # The helper must be called after y / Bmin / Bmax are fully
    # initialised and before the GDP transformation flattens the disjuncts.
    if transformation_type == "bigm":
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
    solver.options["NonConvex"] = 2  # Required for bilinear cooling cost term
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
            print(f"Revenue: {pyo.value(model.Revenue, exception=False)}")
            print(
                f"Outsourcing Cost: {pyo.value(model.OutsourcingCost, exception=False)}"
            )
            print(f"Lateness Cost: {pyo.value(model.LatenessCost, exception=False)}")
            print(
                f"Raw Material Cost: {pyo.value(model.RawMaterialCost, exception=False)}"
            )
            print(
                f"Makespan Penalty: {pyo.value(model.penaltyMS * model.MS, exception=False)}"
            )
            print(f"Cooling Cost: {pyo.value(model.CoolingCost, exception=False)}")
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
