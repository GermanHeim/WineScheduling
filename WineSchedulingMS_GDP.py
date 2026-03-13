"""
Wine Scheduling Optimization Model using Pyomo GDP

This script defines the wine scheduling optimization model using Pyomo GDP
(Generalized Disjunctive Programming) to replace Big-M constraints,
improving readability and potentially numerical stability.
"""

import pyomo.environ as pyo
import pyomo.gdp as gdp
import tomllib
from pyomo.opt import SolverFactory
from utils import export_results


def create_wine_scheduling_model(toml_file="parametersMS.toml"):
    """Create and return the wine scheduling optimization model with GDP"""

    # Load parameters from TOML file
    with open(toml_file, "rb") as f:
        params = tomllib.load(f)

    model = pyo.ConcreteModel(name="WineSchedulingGDP")

    # ========================================
    # SETS
    # ========================================

    # Generate tasks from per-line step definitions
    lines_cfg = params["lines"]
    lines = list(lines_cfg.keys())
    tasks = []
    for line, cfg in lines_cfg.items():
        for step in cfg["steps"]:
            tasks.append(f"{step}{line}")

    # Derived line groups (used for states, ICS/IPS and tc1)
    no_fl_lines = {l for l, cfg in lines_cfg.items() if "Fl" not in cfg["steps"]}
    ist_lines = {l for l, cfg in lines_cfg.items() if "Alm" in cfg["steps"]}
    eco_lines = {l for l, cfg in lines_cfg.items() if cfg.get("ecobulk", False)}

    model.i = pyo.Set(initialize=tasks)
    model.ist = pyo.Set(initialize=[f"Alm{l}" for l in ist_lines])
    model.inst = pyo.Set(initialize=[t for t in tasks if t not in model.ist])
    model.ipst = pyo.Set(initialize=[t for t in tasks if t.startswith("Fr")])
    model.inpst = pyo.Set(initialize=[t for t in tasks if t.startswith(("Fa", "Fl"))])

    model.j = pyo.Set(
        initialize=[
            "inox5000",
            "inox10000",
            "inox17500",
            "inox22000",
            "subte6300",
            "subte7500",
            "subte9000",
            "subte9500",
            "subte12700",
            "iso10000",
            "iso5000",
        ]
    )

    model.JST = pyo.Set(
        initialize=[
            "subte6300",
            "subte7500",
            "subte9000",
            "subte9500",
            "iso10000",
            "iso5000",
        ]
    )

    model.nJST = pyo.Param(initialize=len(model.JST))
    model.inv_nJST = pyo.Param(initialize=1 / len(model.JST))

    # Task-unit pairs (ij)
    ij_data = []
    for task in tasks:
        prefix = ""
        if task.startswith("Fa"):
            prefix = "Fa"
        elif task.startswith("Fl"):
            prefix = "Fl"
        elif task.startswith("Fr"):
            prefix = "Fr"
        elif task.startswith("Alm"):
            prefix = "Alm"

        if (
            prefix in params["templates"]
            and "compatible_units" in params["templates"][prefix]
        ):
            for unit in params["templates"][prefix]["compatible_units"]:
                ij_data.append((task, unit))
    model.ij = pyo.Set(initialize=ij_data, dimen=2)

    model.n = pyo.Set(initialize=list(range(1, params["global"]["n_max"] + 1)))

    # States - built dynamically from line step definitions:
    #   s{l}: raw material
    #   v{l}: after Fa (zero-wait intermediate)
    #   vl{l}: after Fl (NIS intermediate, only when Fl is in steps)
    #   p{l}: final product
    #   dsch: discard / discharge balance state
    states = ["dsch"]
    for line, cfg in lines_cfg.items():
        states.append(f"s{line}")
        states.append(f"v{line}")
        if "Fl" in cfg["steps"]:
            states.append(f"vl{line}")
        states.append(f"p{line}")
    model.s = pyo.Set(initialize=states)

    # States logic
    IPS_data = []
    ICS_data = []
    for task in tasks:
        line = int(task[3:]) if task.startswith("Alm") else int(task[2:])
        if task.startswith("Fa"):
            ICS_data.append((task, f"s{line}"))
            IPS_data.append((task, f"v{line}"))
            IPS_data.append((task, "dsch"))
        elif task.startswith("Fl"):
            ICS_data.append((task, f"v{line}"))
            IPS_data.append((task, f"vl{line}"))
            IPS_data.append((task, "dsch"))
        elif task.startswith("Fr"):
            # No-Fl lines: Fr consumes v{line} directly; others consume vl{line}
            if str(line) in no_fl_lines:
                ICS_data.append((task, f"v{line}"))
            else:
                ICS_data.append((task, f"vl{line}"))
            IPS_data.append((task, f"p{line}"))
        elif task.startswith("Alm"):
            ICS_data.append((task, f"p{line}"))
            IPS_data.append((task, f"p{line}"))
    model.IPS = pyo.Set(initialize=IPS_data, dimen=2)
    model.ICS = pyo.Set(initialize=ICS_data, dimen=2)

    model.SR = pyo.Set(initialize=[f"s{l}" for l in lines])
    model.SP = pyo.Set(initialize=["dsch"] + [f"p{l}" for l in lines])
    model.SI = model.s - model.SP - model.SR
    model.SFISEco = pyo.Set(initialize=[f"p{l}" for l in eco_lines])
    model.SFISBar = pyo.Set(
        initialize=[f"p{l}" for l in lines if l not in eco_lines and l not in ist_lines]
    )
    model.SZW = pyo.Set(initialize=[f"v{l}" for l in lines])
    model.SNIS = pyo.Set(initialize=[f"vl{l}" for l in lines if l not in no_fl_lines])

    tc1_data = []
    for line in lines:
        if line in no_fl_lines:
            tc1_data.append((f"Fa{line}", f"Fr{line}"))
        else:
            tc1_data.append((f"Fa{line}", f"Fl{line}"))
            tc1_data.append((f"Fl{line}", f"Fr{line}"))
    model.tc1 = pyo.Set(initialize=tc1_data, dimen=2)
    tc2_data = [(f"Fr{l}", f"Alm{l}") for l in ist_lines]
    model.tc2 = pyo.Set(initialize=tc2_data, dimen=2)
    model.prd = pyo.Set(initialize=[f"Vino{l}" for l in lines])

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
        prefix = ""
        if task.startswith("Fa"):
            prefix = "Fa"
        elif task.startswith("Fl"):
            prefix = "Fl"
        elif task.startswith("Fr"):
            prefix = "Fr"
        elif task.startswith("Alm"):
            prefix = "Alm"

        if prefix in params["templates"]:
            template = params["templates"][prefix]
            if "alpha" in template:
                model.alpha[task] = template["alpha"]
            model.beta[task] = template["beta"]
            if "compatible_units" in template:
                for unit, limits in template["compatible_units"].items():
                    if (task, unit) in model.ij:
                        model.Bmin[task, unit] = limits[0]
                        model.Bmax[task, unit] = limits[1]

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

    model.penaltyEmptyTank = pyo.Param(initialize=params["global"]["penaltyEmptyTank"])
    model.penaltyAir = pyo.Param(initialize=params["global"]["penaltyAir"])
    model.penaltyMaxUtilization = pyo.Param(
        initialize=params["global"]["penaltyMaxUtilization"]
    )
    model.total_avg_range = pyo.Param(initialize=total_avg_range_value, mutable=False)
    model.inv_total_avg_range = pyo.Param(initialize=1 / total_avg_range_value)

    # Sparse Index Sets
    n_max = max(model.n)

    precedence_pairs = [
        (i, ip, s)
        for i in model.i
        for ip in model.i
        for s in model.s
        if i != ip and (i, s) in model.ICS and (ip, s) in model.IPS
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
    model.bj = pyo.Var(model.i, model.j, model.n, domain=pyo.NonNegativeReals)

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
        for n in model.n:
            model.b[i, n].setub(max_tank_cap * len(model.j))  # Conservative upper bound
            for j in model.j:
                model.bj[i, j, n].setub(max_tank_cap)

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
    model.ProdFinal = pyo.Var(model.SP, domain=pyo.NonNegativeReals)
    model.FueraDeposito = pyo.Var(model.SP, domain=pyo.NonNegativeReals)
    model.MS = pyo.Var(domain=pyo.NonNegativeReals)
    model.JSTsinusar = pyo.Var(domain=pyo.NonNegativeReals)
    model.Freespace = pyo.Var(model.j, domain=pyo.NonNegativeReals)

    ProdFinal_bounds = params["product_ub"]
    for sp in model.SP:
        if sp in ProdFinal_bounds:
            model.ProdFinal[sp].setub(ProdFinal_bounds[sp])

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

    model.g17 = pyo.Constraint(model.SP, rule=lambda m, s: m.ProdFinal[s] >= m.D[s])

    # A01/A02/A01st/A02st: unit-count bounds using disjunct binary_indicator_var as W
    def A01_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            <= model.jMax * active_disjuncts[i, n].binary_indicator_var
        )

    model.A01 = pyo.Constraint(model.inst, model.n, rule=A01_rule)

    def A02_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            >= model.jMin * active_disjuncts[i, n].binary_indicator_var
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
        == m.ProdFinal[s],
    )
    model.A19 = pyo.Constraint(
        model.SP,
        rule=lambda m, s: m.FueraDeposito[s]
        == m.ProdFinal[s]
        - sum(
            m.rhoISprod[i, s] * m.b[i, n] for i in m.ist for n in m.n if (i, s) in m.IPS
        ),
    )
    model.A21 = pyo.Constraint(
        model.ipst, [n_max], rule=lambda m, i, n: m.Tf[i, n] <= m.MS
    )
    model.A22 = pyo.Constraint(
        rule=lambda m: m.JSTsinusar
        == m.nJST
        - sum(m.y[i, j, n] for j in m.JST for n in m.n for i in m.ist if (i, j) in m.ij)
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
    def obj_func(model):
        penalty_unused = model.penaltyEmptyTank * model.inv_nJST * model.JSTsinusar
        penalty_space = (
            model.penaltyAir
            * model.inv_total_avg_range
            * sum(model.Freespace[j] for j in model.JST)
        )
        return model.MS + penalty_unused + penalty_space

    model.OBJ = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # Apply Transformation, set to "hull" or "bigm"
    transformation_type = "hull"
    print(f"Applying GDP {transformation_type} transformation...")
    pyo.TransformationFactory(f"gdp.{transformation_type}").apply_to(model)

    return model


def solve_model(model, solver_name="gurobi", time_limit=3600):
    solver = SolverFactory(solver_name)
    if solver is None:
        print(f"Error: Solver {solver_name} not found")
        return None

    # Set solver options
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = 0.001

    print(f"Solving with {solver_name}...")
    results = solver.solve(model, tee=True)

    if (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
        or results.solver.termination_condition == pyo.TerminationCondition.feasible
    ):
        print(f"Objective: {pyo.value(model.OBJ, exception=False)}")
        export_results(model, "Wine Scheduling GDP", "results_gdp.txt")
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
