"""
Wine Scheduling Optimization Model

This script defines the wine scheduling optimization model using Pyomo,
loading parameters the TOML file.

This model minimizes the makespan of wine production tasks while adhering to
various operational constraints such as task sequencing, storage capacities,
and demand fulfillment.

It also has penalties to encourage efficient use of storage tanks and minimize air
in tanks.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportOptionalOperand=false
# pyright: reportGeneralTypeIssues=false

import pyomo.environ as pyo  # type: ignore
from pyomo.opt import SolverFactory  # type: ignore
import tomllib
from utils import export_results


def create_wine_scheduling_model(toml_file="parametersMS.toml"):
    """Create and return the wine scheduling optimization model"""

    # Load parameters from TOML file
    with open(toml_file, "rb") as f:
        params = tomllib.load(f)

    # Validate TOML structure
    required_sections = [
        "global",
        "lines",
        "templates",
        "products",
        "initial_inventory",
        "product_ub",
    ]

    for section in required_sections:
        if section not in params:
            raise ValueError(f"Missing required section '{section}' in {toml_file}")

    if "idx" not in params["lines"]:
        raise ValueError("Missing 'idx' in [lines] section")

    if not isinstance(params["lines"]["idx"], list):
        raise ValueError("'lines.idx' must be a list")

    # Validate templates
    required_templates = ["Fa", "Fl", "Fr", "Alm"]
    storage_templates = {"Alm"}
    for template in required_templates:
        if template not in params["templates"]:
            raise ValueError(f"Missing template '{template}' in [templates] section")
        template_data = params["templates"][template]
        if template not in storage_templates and "alpha" not in template_data:
            raise ValueError(f"Template '{template}' missing 'alpha'")
        if "beta" not in template_data:
            raise ValueError(f"Template '{template}' missing 'beta'")
        if "compatible_units" not in template_data:
            raise ValueError(f"Template '{template}' missing 'compatible_units'")

    model = pyo.ConcreteModel(name="WineScheduling")

    # ========================================
    # SETS
    # ========================================

    # Generate tasks from lines
    lines = params["lines"]["idx"]
    tasks = []
    for prefix in ["Fa", "Fl", "Fr", "Alm"]:
        for line in lines:
            tasks.append(f"{prefix}{line}")

    # Tasks
    model.i = pyo.Set(initialize=tasks)

    # Storage final tasks
    model.ist = pyo.Set(initialize=["Alm2", "Alm3", "Alm5"])

    # Non-storage tasks
    model.inst = pyo.Set(initialize=[t for t in tasks if t not in model.ist])  # type: ignore

    # Pre-storage tasks
    model.ipst = pyo.Set(initialize=[t for t in tasks if t.startswith("Fr")])

    # Non-pre-storage non-final tasks
    model.inpst = pyo.Set(initialize=[t for t in tasks if t.startswith(("Fa", "Fl"))])

    # Units
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

    # Storage units
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

    # Number of storage units
    model.nJST = pyo.Param(initialize=len(model.JST))

    # Inverse for normalization
    model.inv_nJST = pyo.Param(initialize=1 / len(model.JST))  # type: ignore

    # Task-unit pairs (ij)
    ij_data = []
    for task in tasks:
        if task.startswith("Fa"):
            prefix = "Fa"
        elif task.startswith("Fl"):
            prefix = "Fl"
        elif task.startswith("Fr"):
            prefix = "Fr"
        elif task.startswith("Alm"):
            prefix = "Alm"
        else:
            continue

        if (
            prefix in params["templates"]
            and "compatible_units" in params["templates"][prefix]
        ):
            for unit in params["templates"][prefix]["compatible_units"]:
                ij_data.append((task, unit))
    model.ij = pyo.Set(initialize=ij_data, dimen=2)

    # Events
    model.n = pyo.Set(initialize=list(range(1, params["global"]["n_max"] + 1)))

    # States
    model.s = pyo.Set(
        initialize=[
            "s1",
            "v1",
            "vl1",
            "p1",
            "s2",
            "v2",
            "vl2",
            "p2",
            "s3",
            "v3",
            "vl3",
            "p3",
            "s4",
            "v4",
            "vl4",
            "p4",
            "s5",
            "v5",
            "vl5",
            "p5",
            "s6",
            "v6",
            "vl6",
            "p6",
            "dsch",
        ]
    )

    # States produced by task
    IPS_data = []
    ICS_data = []
    for task in tasks:
        if task.startswith("Alm"):
            line = int(task[3:])
        else:
            line = int(task[2:])
        if task.startswith("Fa"):
            ICS_data.append((task, f"s{line}"))
            IPS_data.append((task, f"v{line}"))
            IPS_data.append((task, "dsch"))
        elif task.startswith("Fl"):
            ICS_data.append((task, f"v{line}"))
            IPS_data.append((task, f"vl{line}"))
            IPS_data.append((task, "dsch"))
        elif task.startswith("Fr"):
            ICS_data.append((task, f"vl{line}"))
            IPS_data.append((task, f"p{line}"))
        elif task.startswith("Alm"):
            ICS_data.append((task, f"p{line}"))
            IPS_data.append((task, f"p{line}"))
    model.IPS = pyo.Set(initialize=IPS_data, dimen=2)

    # States consumed by task
    model.ICS = pyo.Set(initialize=ICS_data, dimen=2)

    # Raw material states
    model.SR = pyo.Set(initialize=["s1", "s2", "s3", "s4", "s5", "s6"])

    # Final product states
    model.SP = pyo.Set(initialize=["dsch", "p1", "p2", "p3", "p4", "p5", "p6"])

    # Intermediate states
    model.SI = model.s - model.SP - model.SR  # type: ignore

    # Final states that can be stored in ecobulk
    model.SFISEco = pyo.Set(initialize=["p2", "p3", "p5"])

    # Final states stored only in barrels
    model.SFISBar = pyo.Set(initialize=["p1", "p4", "p6"])

    # Zero wait states
    model.SZW = pyo.Set(initialize=["v1", "v2", "v3", "v4", "v5", "v6"])

    # No intermediate storage states (but can wait)
    model.SNIS = pyo.Set(initialize=["vl1", "vl2", "vl3", "vl4", "vl5", "vl6"])

    # One-to-one task pairs
    tc1_data = [
        ("Fa1", "Fl1"),
        ("Fl1", "Fr1"),
        ("Fa2", "Fl2"),
        ("Fl2", "Fr2"),
        ("Fa3", "Fl3"),
        ("Fl3", "Fr3"),
        ("Fa4", "Fl4"),
        ("Fl4", "Fr4"),
        ("Fa5", "Fl5"),
        ("Fl5", "Fr5"),
        ("Fa6", "Fl6"),
        ("Fl6", "Fr6"),
    ]
    model.tc1 = pyo.Set(initialize=tc1_data, dimen=2)

    # One-to-one storage pairs
    tc2_data = [("Fr2", "Alm2"), ("Fr3", "Alm3"), ("Fr5", "Alm5")]
    model.tc2 = pyo.Set(initialize=tc2_data, dimen=2)

    # Products
    model.prd = pyo.Set(
        initialize=["Vino1", "Vino2", "Vino3", "Vino4", "Vino5", "Vino6"]
    )

    # ========================================
    # PARAMETERS
    # ========================================

    model.iMax = pyo.Param(initialize=1)
    model.jMax = pyo.Param(initialize=1)
    model.jMin = pyo.Param(initialize=1)
    model.iMaxST = pyo.Param(initialize=1)
    model.jMaxST = pyo.Param(initialize=2)
    model.jMinST = pyo.Param(initialize=1)

    # =1 if the user wants to force wines using ecobulk to use some kind of tank for final storage
    # =0 if it is optimized
    model.MustUseEcobulk = pyo.Param(initialize=0)

    # Initial state quantities
    model.ST0 = pyo.Param(model.s, initialize=params["initial_inventory"], default=0)

    # Maximum storage capacity
    model.STmax = pyo.Param(model.s, initialize=0)

    # Processing time parameters (Loaded from data)
    model.alpha = pyo.Param(model.i, mutable=True, default=0)
    model.beta = pyo.Param(model.i, mutable=True, default=0)

    # Batch size limits (Loaded from data)
    model.Bmin = pyo.Param(model.ij, mutable=True, default=0)
    model.Bmax = pyo.Param(model.ij, mutable=True, default=0)

    # Load alpha, beta, Bmin, Bmax from templates
    for task in model.i:  # type: ignore
        if task is not None:
            if task.startswith("Fa"):
                prefix = "Fa"
            elif task.startswith("Fl"):
                prefix = "Fl"
            elif task.startswith("Fr"):
                prefix = "Fr"
            elif task.startswith("Alm"):
                prefix = "Alm"
            else:
                continue

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

    # Compute total_avg_range
    total = 0.0
    for j in model.JST:
        tank_range_sum = 0.0
        tank_count = 0
        for i in model.ist:
            if (i, j) in model.ij:
                bmax_val = pyo.value(model.Bmax[i, j])
                bmin_val = pyo.value(model.Bmin[i, j])
                tank_range_sum += bmax_val - bmin_val
                tank_count += 1
        if tank_count == 0:
            raise ValueError(f"No storage tasks assigned to storage unit {j}")
        total += tank_range_sum / tank_count
    total_avg_range_value = total if total > 0.0 else 1.0

    # Horizon
    model.H = pyo.Param(initialize=params["global"]["H"])

    # Big M
    model.M = pyo.Param(initialize=params["global"]["H"])

    # State consumption/production coefficients (Loaded from TOML)
    model.rhoIScons = pyo.Param(model.i, model.s, mutable=True, default=0)
    model.rhoISprod = pyo.Param(model.i, model.s, mutable=True, default=0)

    # Load rhoIScons and rhoISprod from TOML rho section
    if "rho" in params:
        for task_key, rho_data in params["rho"].items():
            if task_key in model.i:  # type: ignore
                if "rhoIScons" in rho_data:
                    for state, value in rho_data["rhoIScons"].items():
                        model.rhoIScons[task_key, state] = value
                if "rhoISprod" in rho_data:
                    for state, value in rho_data["rhoISprod"].items():
                        model.rhoISprod[task_key, state] = value

    # Demand
    demand_data = {k: v["demand"] for k, v in params["products"].items()}
    model.D = pyo.Param(model.s, initialize=demand_data, default=0)

    # Penalties
    # Penalty imposed for leaving tanks empty
    model.penaltyEmptyTank = pyo.Param(initialize=params["global"]["penaltyEmptyTank"])
    # Penalty imposed for leaving air in the tanks
    model.penaltyAir = pyo.Param(initialize=params["global"]["penaltyAir"])
    # Penalty imposed to encourage the maximum use of deposits
    model.penaltyMaxUtilization = pyo.Param(
        initialize=params["global"]["penaltyMaxUtilization"]
    )

    # Total average tank range for normalization in objective function
    model.total_avg_range = pyo.Param(initialize=total_avg_range_value, mutable=False)

    # Inverse for normalization
    model.inv_total_avg_range = pyo.Param(initialize=1 / total_avg_range_value)

    # ========================================
    # SPARSE INDEX SETS FOR CONSTRAINTS
    # ========================================

    n_max = max(model.n)

    # Precedence pairs for g11, g12, g13, A16
    precedence_pairs = [
        (i, ip, s)
        for i in model.i
        for ip in model.i
        for s in model.s
        if i != ip and (i, s) in model.ICS and (ip, s) in model.IPS
    ]
    model.PrecedencePairs = pyo.Set(initialize=precedence_pairs, dimen=3)

    # Zero wait precedence pairs
    zw_precedence_pairs = [
        (i, ip, s) for i, ip, s in precedence_pairs if s in model.SZW
    ]
    model.ZWPrecedencePairs = pyo.Set(initialize=zw_precedence_pairs, dimen=3)

    # No intermediate storage precedence pairs
    nis_precedence_pairs = [
        (i, ip, s) for i, ip, s in precedence_pairs if s in model.SNIS
    ]
    model.NISPrecedencePairs = pyo.Set(initialize=nis_precedence_pairs, dimen=3)

    # Ecobulk precedence pairs
    eco_precedence_pairs = [
        (i, ip, s) for i, ip, s in precedence_pairs if s in model.SFISEco
    ]
    model.EcoPrecedencePairs = pyo.Set(initialize=eco_precedence_pairs, dimen=3)

    # Precedence pairs with n for g11, g12, g13
    precedence_pairs_with_n = [
        (i, ip, s, n) for i, ip, s in precedence_pairs for n in model.n if n < n_max
    ]
    model.PrecedencePairsWithN = pyo.Set(initialize=precedence_pairs_with_n, dimen=4)

    # Zero wait precedence pairs with n
    zw_precedence_pairs_with_n = [
        (i, ip, s, n) for i, ip, s in zw_precedence_pairs for n in model.n if n < n_max
    ]
    model.ZWPrecedencePairsWithN = pyo.Set(
        initialize=zw_precedence_pairs_with_n, dimen=4
    )

    # No intermediate storage precedence pairs with n
    nis_precedence_pairs_with_n = [
        (i, ip, s, n) for i, ip, s in nis_precedence_pairs for n in model.n if n < n_max
    ]
    model.NISPrecedencePairsWithN = pyo.Set(
        initialize=nis_precedence_pairs_with_n, dimen=4
    )

    # Ecobulk precedence pairs with n
    eco_precedence_pairs_with_n = [
        (i, ip, s, n) for i, ip, s in eco_precedence_pairs for n in model.n if n < n_max
    ]
    model.EcoPrecedencePairsWithN = pyo.Set(
        initialize=eco_precedence_pairs_with_n, dimen=4
    )

    # For h04: material balance first event
    h04_indices = [(s, 1) for s in model.s]
    model.h04_indices = pyo.Set(initialize=h04_indices, dimen=2)

    # For h03: material balance subsequent events
    h03_indices = [(s, n) for s in model.s for n in model.n if n > 1]
    model.h03_indices = pyo.Set(initialize=h03_indices, dimen=2)

    # For h05: processing time normal tasks
    h05_indices = [(i, n) for i in model.inst for n in model.n]
    model.h05_indices = pyo.Set(initialize=h05_indices, dimen=2)

    # For g06: task sequencing
    g06_indices = [(i, n) for i in model.i for n in model.n if n < n_max]
    model.g06_indices = pyo.Set(initialize=g06_indices, dimen=2)

    # For A01: max units per normal task
    A01_indices = [(i, n) for i in model.inst for n in model.n]
    model.A01_indices = pyo.Set(initialize=A01_indices, dimen=2)

    # For A02: min units per normal task
    A02_indices = [(i, n) for i in model.inst for n in model.n]
    model.A02_indices = pyo.Set(initialize=A02_indices, dimen=2)

    # For A01st: max units per storage task
    A01st_indices = [(i, n) for i in model.ist for n in model.n]
    model.A01st_indices = pyo.Set(initialize=A01st_indices, dimen=2)

    # For A02st: min units per storage task
    A02st_indices = [(i, n) for i in model.ist for n in model.n]
    model.A02st_indices = pyo.Set(initialize=A02st_indices, dimen=2)

    # For A04: total batch
    A04_indices = [(i, n) for i in model.i for n in model.n]
    model.A04_indices = pyo.Set(initialize=A04_indices, dimen=2)

    # For A05b: max batch size
    A05b_indices = [
        (i, j, n)
        for i in model.i
        for j in model.j
        for n in model.n
        if (i, j) in model.ij
    ]
    model.A05b_indices = pyo.Set(initialize=A05b_indices, dimen=3)

    # For A06: max activations normal
    A06_indices = [i for i in model.inst]
    model.A06_indices = pyo.Set(initialize=A06_indices, dimen=1)

    # For A06st: max activations storage
    A06st_indices = [i for i in model.ist]
    model.A06st_indices = pyo.Set(initialize=A06st_indices, dimen=1)

    # For A07: storage task duration min
    A07_indices = [(i, n) for i in model.ist for n in model.n]
    model.A07_indices = pyo.Set(initialize=A07_indices, dimen=2)

    # For A08: storage task must last until horizon
    A08_indices = [(i, n) for i in model.ist for n in model.n]
    model.A08_indices = pyo.Set(initialize=A08_indices, dimen=2)

    # For A09: unit event sequencing
    A09_indices = [(j, n) for j in model.j for n in model.n if n < n_max]
    model.A09_indices = pyo.Set(initialize=A09_indices, dimen=2)

    # For A10: unit event duration
    A10_indices = [(j, n) for j in model.j for n in model.n]
    model.A10_indices = pyo.Set(initialize=A10_indices, dimen=2)

    # For A11b: task-unit start sync
    A11b_indices = [
        (i, j, n)
        for i in model.i
        for j in model.j
        for n in model.n
        if (i, j) in model.ij
    ]
    model.A11b_indices = pyo.Set(initialize=A11b_indices, dimen=3)

    # For A12a: task-unit end sync
    A12a_indices = [
        (i, j, n)
        for i in model.i
        for j in model.j
        for n in model.n
        if (i, j) in model.ij
    ]
    model.A12a_indices = pyo.Set(initialize=A12a_indices, dimen=3)

    # For A12b: task-unit end sync
    A12b_indices = [
        (i, j, n)
        for i in model.i
        for j in model.j
        for n in model.n
        if (i, j) in model.ij
    ]
    model.A12b_indices = pyo.Set(initialize=A12b_indices, dimen=3)

    # For A13b: one-to-one storage pairs mandatory
    A13b_indices = [
        (i, ip, n)
        for i in model.i
        for ip in model.i
        for n in model.n
        if (i, ip) in model.tc2 and model.MustUseEcobulk == 1 and n < n_max
    ]
    model.A13b_indices = pyo.Set(initialize=A13b_indices, dimen=3)

    # For A13c: one-to-one storage pairs optional
    A13c_indices = [
        (i, ip, n)
        for i in model.i
        for ip in model.i
        for n in model.n
        if (i, ip) in model.tc2 and model.MustUseEcobulk != 1 and n < n_max
    ]
    model.A13c_indices = pyo.Set(initialize=A13c_indices, dimen=3)

    # For A14: storage task persistence (recursive chain: n+1 >= n)
    A14_indices = [(i, n) for i in model.ist for n in model.n if n < n_max]
    model.A14_indices = pyo.Set(initialize=A14_indices, dimen=2)

    # For A15: storage unit persistence (recursive chain: n+1 >= n)
    A15_indices = [
        (i, j, n)
        for i in model.ist
        for j in model.j
        for n in model.n
        if (i, j) in model.ij and n < n_max
    ]
    model.A15_indices = pyo.Set(initialize=A15_indices, dimen=3)

    # For A17: must use deposit for ecobulk
    A17_indices = [s for s in model.SFISEco if model.MustUseEcobulk == 1]
    model.A17_indices = pyo.Set(initialize=A17_indices, dimen=1)

    # For A18: final production calculation
    A18_indices = [(s, n) for s in model.SP for n in model.n if n == n_max]
    model.A18_indices = pyo.Set(initialize=A18_indices, dimen=2)

    # For A21: makespan
    A21_indices = [(i, n) for i in model.ipst for n in model.n if n == n_max]
    model.A21_indices = pyo.Set(initialize=A21_indices, dimen=2)

    # For A23: free space
    A23_indices = [(j, i) for j in model.JST for i in model.ist if (i, j) in model.ij]
    model.A23_indices = pyo.Set(initialize=A23_indices, dimen=2)

    # For A24: free space
    A24_indices = [(j, i) for j in model.JST for i in model.ist if (i, j) in model.ij]
    model.A24_indices = pyo.Set(initialize=A24_indices, dimen=2)

    # ========================================
    # VARIABLES
    # ========================================

    # Binary variables
    model.W = pyo.Var(model.i, model.n, domain=pyo.Binary)
    model.y = pyo.Var(model.ij, model.n, domain=pyo.Binary)

    # Positive variables
    model.b = pyo.Var(model.i, model.n, domain=pyo.NonNegativeReals)
    model.bj = pyo.Var(model.i, model.j, model.n, domain=pyo.NonNegativeReals)
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

    # Upper bounds on final production
    ProdFinal_bounds = params["product_ub"]
    for sp in model.SP:  # type: ignore
        if sp in ProdFinal_bounds:
            model.ProdFinal[sp].setub(ProdFinal_bounds[sp])

    # ========================================
    # CONSTRAINTS
    # ========================================

    # Material balance - first event (h04)
    def h04_rule(model, s, n):
        consumed = sum(
            model.rhoIScons[i, s] * model.b[i, n]
            for i in model.i
            if (i, s) in model.ICS
        )
        return model.ST[s, n] == model.ST0[s] + consumed

    model.h04 = pyo.Constraint(model.h04_indices, rule=h04_rule)

    # Material balance - subsequent events (h03)
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

    model.h03 = pyo.Constraint(model.h03_indices, rule=h03_rule)

    # Processing time for normal tasks (h05)
    def h05_rule(model, i, n):
        return (
            model.Tf[i, n]
            == model.Ts[i, n]
            + model.alpha[i] * model.W[i, n]
            + model.beta[i] * model.b[i, n]
        )

    model.h05 = pyo.Constraint(model.h05_indices, rule=h05_rule)

    # Task sequencing (g06)
    def g06_rule(model, i, n):
        return model.Ts[i, n + 1] >= model.Tf[i, n]

    model.g06 = pyo.Constraint(model.g06_indices, rule=g06_rule)

    # Task precedence (g11)
    def g11_rule(model, i, ip, s, n):
        return model.Ts[i, n + 1] >= model.Tf[ip, n] - model.M * (1 - model.W[ip, n])

    model.g11 = pyo.Constraint(model.PrecedencePairsWithN, rule=g11_rule)

    # Zero wait constraints (g12)
    def g12_rule(model, i, ip, s, n):
        return model.Ts[i, n + 1] <= model.Tf[ip, n] + model.M * (
            2 - model.W[ip, n] - model.W[i, n + 1]
        )

    model.g12 = pyo.Constraint(model.ZWPrecedencePairsWithN, rule=g12_rule)

    # No intermediate storage constraints (g13)
    def g13_rule(model, i, ip, s, n):
        return model.Ts[i, n + 1] <= model.Tf[ip, n] + model.M * (
            2 - model.W[ip, n] - model.W[i, n + 1]
        )

    model.g13 = pyo.Constraint(model.NISPrecedencePairsWithN, rule=g13_rule)

    # Unit capacity constraint (h09)
    def h09_rule(model, j):
        total_time = sum(
            model.alpha[i] * model.y[i, j, n] + model.beta[i] * model.bj[i, j, n]
            for i in model.inst
            for n in model.n
            if (i, j) in model.ij
        )
        return total_time <= model.H

    model.h09 = pyo.Constraint(model.j, rule=h09_rule)

    # Zero wait state inventory (h14)
    def h14_rule(model, s, n):
        return model.ST[s, n] == 0

    model.h14 = pyo.Constraint(model.SZW, model.n, rule=h14_rule)

    # No intermediate storage state inventory (h15)
    def h15_rule(model, s, n):
        return model.ST[s, n] == 0

    model.h15 = pyo.Constraint(model.SNIS, model.n, rule=h15_rule)

    # Zero wait final state (h16)
    def h16_rule(model, s):
        n_max = max(model.n)
        produced = sum(
            model.rhoISprod[i, s] * model.b[i, n_max]
            for i in model.i
            if (i, s) in model.IPS
        )
        return model.ST[s, n_max] + produced == 0

    model.h16 = pyo.Constraint(model.SZW, rule=h16_rule)

    # Demand constraint (g17)
    def g17_rule(model, s):
        return model.ProdFinal[s] >= model.D[s]

    model.g17 = pyo.Constraint(model.SP, rule=g17_rule)

    # Maximum units per normal task (A01)
    def A01_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            <= model.jMax * model.W[i, n]
        )

    model.A01 = pyo.Constraint(model.A01_indices, rule=A01_rule)

    # Minimum units per normal task (A02)
    def A02_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            >= model.jMin * model.W[i, n]
        )

    model.A02 = pyo.Constraint(model.A02_indices, rule=A02_rule)

    # Maximum units per storage task (A01st)
    def A01st_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            <= model.jMaxST * model.W[i, n]
        )

    model.A01st = pyo.Constraint(model.A01st_indices, rule=A01st_rule)

    # Minimum units per storage task (A02st)
    def A02st_rule(model, i, n):
        return (
            sum(model.y[i, j, n] for j in model.j if (i, j) in model.ij)
            >= model.jMinST * model.W[i, n]
        )

    model.A02st = pyo.Constraint(model.A02st_indices, rule=A02st_rule)

    # Each unit can only be used by one task per event (A03)
    def A03_rule(model, j, n):
        return sum(model.y[i, j, n] for i in model.i if (i, j) in model.ij) <= 1

    model.A03 = pyo.Constraint(model.j, model.n, rule=A03_rule)

    # Total batch is sum of batches in all units (A04)
    def A04_rule(model, i, n):
        return model.b[i, n] == sum(
            model.bj[i, j, n] for j in model.j if (i, j) in model.ij
        )

    model.A04 = pyo.Constraint(model.A04_indices, rule=A04_rule)

    # Minimum batch size (A05a)
    def A05a_rule(model, i, j, n):
        return model.Bmin[i, j] * model.y[i, j, n] <= model.bj[i, j, n]

    model.A05a = pyo.Constraint(model.ij, model.n, rule=A05a_rule)

    # Maximum batch size (A05b)
    def A05b_rule(model, i, j, n):
        return model.Bmax[i, j] * model.y[i, j, n] >= model.bj[i, j, n]

    model.A05b = pyo.Constraint(model.A05b_indices, rule=A05b_rule)

    # Maximum activations per normal task (A06)
    def A06_rule(model, i):
        return sum(model.W[i, n] for n in model.n) <= model.iMax

    model.A06 = pyo.Constraint(model.A06_indices, rule=A06_rule)

    # Maximum activations per storage task (A06st)
    # With persistence (A14), W[i,n] is non-decreasing, so checking only the
    # final event W[i, n_max] is sufficient to limit activations.
    def A06st_rule(model, i):
        return model.W[i, n_max] <= model.iMaxST

    model.A06st = pyo.Constraint(model.A06st_indices, rule=A06st_rule)

    # Storage task duration minimum (A07)
    def A07_rule(model, i, n):
        return model.Tf[i, n] >= model.Ts[i, n]

    model.A07 = pyo.Constraint(model.A07_indices, rule=A07_rule)

    # Storage task must last until horizon (A08)
    def A08_rule(model, i, n):
        return model.Tf[i, n] >= model.H * model.W[i, n]

    model.A08 = pyo.Constraint(model.A08_indices, rule=A08_rule)

    # Unit event sequencing (A09)
    def A09_rule(model, j, n):
        return model.Tsj[j, n + 1] >= model.Tfj[j, n]

    model.A09 = pyo.Constraint(model.A09_indices, rule=A09_rule)

    # Unit event duration (A10)
    def A10_rule(model, j, n):
        return model.Tfj[j, n] >= model.Tsj[j, n]

    model.A10 = pyo.Constraint(model.A10_indices, rule=A10_rule)

    # Task-unit start time synchronization (A11a, A11b)
    def A11a_rule(model, i, j, n):
        return model.Tsj[j, n] <= model.Ts[i, n] + model.H * (1 - model.y[i, j, n])

    model.A11a = pyo.Constraint(model.ij, model.n, rule=A11a_rule)

    def A11b_rule(model, i, j, n):
        return model.Tsj[j, n] >= model.Ts[i, n] - model.H * (1 - model.y[i, j, n])

    model.A11b = pyo.Constraint(model.A11b_indices, rule=A11b_rule)

    # Task-unit end time synchronization (A12a, A12b)
    def A12a_rule(model, i, j, n):
        return model.Tfj[j, n] <= model.Tf[i, n] + model.H * (1 - model.y[i, j, n])

    model.A12a = pyo.Constraint(model.A12a_indices, rule=A12a_rule)

    def A12b_rule(model, i, j, n):
        return model.Tfj[j, n] >= model.Tf[i, n] - model.H * (1 - model.y[i, j, n])

    model.A12b = pyo.Constraint(model.A12b_indices, rule=A12b_rule)

    # One-to-one task pairs (A13a)
    def A13a_rule(model, i, ip, n):
        return model.W[ip, n + 1] == model.W[i, n]

    model.A13a = pyo.Constraint(
        model.tc1, [n for n in model.n if n < n_max], rule=A13a_rule
    )

    # One-to-one storage pairs - mandatory (A13b)
    def A13b_rule(model, i, ip, n):
        return model.W[ip, n + 1] == model.W[i, n]

    model.A13b = pyo.Constraint(model.A13b_indices, rule=A13b_rule)

    # One-to-one storage pairs - optional (A13c)
    def A13c_rule(model, i, ip, n):
        return model.W[ip, n + 1] <= model.W[i, n]

    model.A13c = pyo.Constraint(model.A13c_indices, rule=A13c_rule)

    # Storage task persistence (A14) - recursive chain
    def A14_rule(model, i, n):
        return model.W[i, n + 1] >= model.W[i, n]

    model.A14 = pyo.Constraint(model.A14_indices, rule=A14_rule)

    # Storage unit persistence (A15) - recursive chain
    def A15_rule(model, i, j, n):
        return model.y[i, j, n + 1] >= model.y[i, j, n]

    model.A15 = pyo.Constraint(model.A15_indices, rule=A15_rule)

    # Ecobulk storage constraints (A16)
    def A16_rule(model, i, ip, s, n):
        return model.Ts[i, n + 1] <= model.Tf[ip, n] + model.M * (
            2 - model.W[ip, n] - model.W[i, n + 1]
        )

    model.A16 = pyo.Constraint(model.EcoPrecedencePairsWithN, rule=A16_rule)

    # Must use deposit for ecobulk (A17)
    def A17_rule(model, s):
        return (
            sum(model.W[i, n] for i in model.i for n in model.n if (i, s) in model.ICS)
            >= 1
        )

    model.A17 = pyo.Constraint(model.A17_indices, rule=A17_rule)

    # Final production calculation (A18)
    def A18_rule(model, s, n):
        produced = sum(
            model.rhoISprod[i, s] * model.b[i, n]
            for i in model.i
            if (i, s) in model.IPS
        )
        return model.ST[s, n] + produced == model.ProdFinal[s]

    model.A18 = pyo.Constraint(model.A18_indices, rule=A18_rule)

    # Out of deposit storage (A19)
    def A19_rule(model, s):
        in_deposit = sum(
            model.rhoISprod[i, s] * model.b[i, n]
            for i in model.ist
            for n in model.n
            if (i, s) in model.IPS
        )
        return model.FueraDeposito[s] == model.ProdFinal[s] - in_deposit

    model.A19 = pyo.Constraint(model.SP, rule=A19_rule)

    # Makespan constraint (A21)
    def A21_rule(model, i, n):
        return model.Tf[i, n] <= model.MS

    model.A21 = pyo.Constraint(model.A21_indices, rule=A21_rule)

    # Unused storage units (A22)
    def A22_rule(model):
        used_units = sum(
            model.y[i, j, n]
            for j in model.JST
            for n in model.n
            for i in model.ist
            if (i, j) in model.ij
        )
        return model.JSTsinusar == model.nJST - used_units

    model.A22 = pyo.Constraint(rule=A22_rule)

    # Free space in units (A23, A24)
    def A23_rule(model, j, i):
        used = sum(model.bj[i, j, n] for n in model.n)
        used_flag = sum(model.y[i, j, n] for n in model.n)
        return model.Freespace[j] <= model.Bmax[i, j] - used + model.Bmax[i, j] * (
            1 - used_flag
        )

    model.A23 = pyo.Constraint(model.A23_indices, rule=A23_rule)

    def A24_rule(model, j, i):
        used = sum(model.bj[i, j, n] for n in model.n)
        used_flag = sum(model.y[i, j, n] for n in model.n)
        return model.Freespace[j] >= model.Bmax[i, j] - used - model.Bmax[i, j] * (
            1 - used_flag
        )

    model.A24 = pyo.Constraint(model.A24_indices, rule=A24_rule)

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

    # print("=" * 80)
    # print("CONSTRAINT COUNTS")
    # print("=" * 80)
    # constraint_counts = {}
    # total_constraints = 0

    # for constraint_name in dir(model):
    #     obj = getattr(model, constraint_name)
    #     if isinstance(obj, pyo.Constraint):
    #         count = len(obj)
    #         constraint_counts[constraint_name] = count
    #         total_constraints += count

    # for name in sorted(constraint_counts.keys()):
    #     print(f"{name:10s}: {constraint_counts[name]:6d}")

    # print("=" * 80)
    # print(f"{'TOTAL':10s}: {total_constraints:6d}")
    # print("=" * 80)

    return model


def solve_model(model, solver_name="gurobi", time_limit=3600):
    """Solve the model using GUROBI"""

    solver = SolverFactory(solver_name)

    if solver is None:
        print(f"Error: Solver {solver_name} not found.")
        return None

    # Set solver options
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = 0.0  # optcr = 0 in GAMS

    print(f"\nSolving model with {solver_name}...")
    print(f"Time limit: {time_limit} seconds")
    print(f"Horizon: {pyo.value(model.H)} hours")

    # Write LP file for debugging
    # model.write("wine_scheduling_model.lp", io_options={"symbolic_solver_labels": True})
    # print("LP file written to wine_scheduling_model.lp")

    print(f"Vars: {model.nvariables()}, Cons: {model.nconstraints()}")

    # Solve
    results = solver.solve(model, tee=True)

    # Display results
    print("\n" + "=" * 80)
    print("SOLUTION STATUS")
    print("=" * 80)
    print(f"Solver Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")

    # If infeasible or unbounded, try to diagnose
    if (
        results.solver.termination_condition
        == pyo.TerminationCondition.infeasibleOrUnbounded
    ):
        print("\nModel is infeasible or unbounded.")
        print("Trying to determine which by disabling dual reductions...")

        # Disable dual reductions to distinguish infeasible from unbounded
        solver.options["DualReductions"] = 0
        results2 = solver.solve(model, tee=False)

        if results2.solver.termination_condition == pyo.TerminationCondition.infeasible:
            print("Model is INFEASIBLE.")

            # Compute IIS to identify conflicting constraints
            print("\nComputing Irreducible Inconsistent Subsystem (IIS)...")
            try:
                solver.options["IISMethod"] = 0
                model.write("infeasible_model.lp")
                # Need to get the GUROBI model object directly
                from gurobipy import read

                gurobi_model = read("infeasible_model.lp")
                gurobi_model.computeIIS()
                gurobi_model.write("model_iis.ilp")
            except Exception as e:
                print(f"Could not compute IIS: {e}")
        elif (
            results2.solver.termination_condition == pyo.TerminationCondition.unbounded
        ):
            print("Model is UNBOUNDED.")

        # Restore original options
        del solver.options["DualReductions"]
        results = results2

    if (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
        or results.solver.termination_condition == pyo.TerminationCondition.feasible
    ):
        print(f"\nObjective Value (Makespan): {pyo.value(model.OBJ):.2f}")
        print(
            f"Makespan: {pyo.value(model.MS):.2f} hours ({pyo.value(model.MS) / 24:.2f} days)"
        )

        # Display production results
        print("=" * 80)
        print("FINAL PRODUCTION")
        print("=" * 80)
        for s in model.SP:
            if pyo.value(model.ProdFinal[s]) > 0.01:
                print(
                    f"{s}: {pyo.value(model.ProdFinal[s]):.2f} L (Demand: {model.D[s]:.2f} L)"
                )
                if pyo.value(model.FueraDeposito[s]) > 0.01:
                    print(
                        f"  -> Out of deposit: {pyo.value(model.FueraDeposito[s]):.2f} L"
                    )

        # Display task schedule
        print("\n" + "=" * 80)
        print("TASK SCHEDULE")
        print("=" * 80)
        for i in model.i:
            for n in model.n:
                if pyo.value(model.W[i, n]) > 0.5:
                    print(f"\nTask {i}, Event {n}:")
                    print(
                        f"  Start: {pyo.value(model.Ts[i, n]):.2f} h, End: {pyo.value(model.Tf[i, n]):.2f} h"
                    )
                    print(f"  Batch: {pyo.value(model.b[i, n]):.2f} L")
                    print("  Units:", end=" ")
                    for j in model.j:
                        if (i, j) in model.ij and pyo.value(model.y[i, j, n]) > 0.5:
                            print(
                                f"{j} ({pyo.value(model.bj[i, j, n]):.2f} L)", end=" "
                            )
                    print()
        print("\n" + "=" * 80)
    else:
        print("\nNo optimal solution found!")

    return results


def main():
    print("=" * 80)
    print("Wine Scheduling Optimization")
    print("=" * 80)

    print("Creating optimization model...")
    model = create_wine_scheduling_model()
    print("Model created successfully")
    print(f"  - {len(model.i)} tasks")
    print(f"  - {len(model.j)} units")
    print(f"  - {len(model.n)} events")
    print(f"  - {len(model.s)} states")

    results = solve_model(model, solver_name="gurobi", time_limit=3600)
    if results and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
        or results.solver.termination_condition == pyo.TerminationCondition.feasible
    ):
        export_results(model, "Wine Scheduling Makespan Minimization", "results.txt")

    return model, results


if __name__ == "__main__":
    model, results = main()
