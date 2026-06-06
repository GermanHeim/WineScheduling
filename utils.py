"""
Utility functions for Wine Scheduling Optimization models.
"""

import json
from datetime import datetime

import pyomo.environ as pyo  # type: ignore
import pyomo.gdp as gdp  # type: ignore


DEFAULT_WARMSTART_VARS = (
    "W",
    "y",
    "b",
    "bj",
    "Ts",
    "Tf",
    "Tsj",
    "Tfj",
    "ST",
    "MS",
    "NumBarr",
    "NumJar",
)


def idx_to_key(idx):
    if idx is None:
        return "__scalar__"
    if isinstance(idx, tuple):
        return json.dumps(list(idx))
    return json.dumps([idx])


def key_to_idx(key):
    if key == "__scalar__":
        return None
    parts = json.loads(key)
    return tuple(parts) if len(parts) > 1 else parts[0]


def extract_warm_start(model, var_names=DEFAULT_WARMSTART_VARS):
    """
    Snapshot primal values of named variables plus all GDP disjunct
    indicator variables. Returns a JSON-serializable dict suitable for
    passing to apply_warm_start on a structurally similar model.
    """
    snapshot = {"vars": {}, "indicators": {}}
    for name in var_names:
        comp = model.find_component(name)
        if comp is None:
            continue
        entries = {}
        if comp.is_indexed():
            for idx in comp:
                v = comp[idx].value
                if v is not None:
                    entries[idx_to_key(idx)] = float(v)
        else:
            v = comp.value
            if v is not None:
                entries["__scalar__"] = float(v)
        if entries:
            snapshot["vars"][name] = entries

    for disj in model.component_objects(gdp.Disjunct, descend_into=True):
        for idx in disj:
            try:
                ind = disj[idx].binary_indicator_var
            except AttributeError:
                continue
            v = ind.value
            if v is None:
                continue
            snapshot["indicators"][f"{disj.name}::{idx_to_key(idx)}"] = float(v)
    return snapshot


def save_warm_start(snapshot, path):
    with open(path, "w") as f:
        json.dump(snapshot, f)
    print(f"Warm start saved to {path}")


def load_warm_start(path):
    with open(path) as f:
        return json.load(f)


def apply_warm_start(model, snapshot):
    """
    Set .value on model variables matching snapshot. Missing components
    or indices are silently skipped,since Gurobi accepts a partial start.
    Returns (applied, skipped).
    """
    applied = skipped = 0

    for name, entries in snapshot.get("vars", {}).items():
        comp = model.find_component(name)
        if comp is None:
            skipped += len(entries)
            continue
        for key, val in entries.items():
            if key == "__scalar__":
                try:
                    comp.set_value(val)
                    applied += 1
                except Exception:
                    skipped += 1
                continue
            idx = key_to_idx(key)
            try:
                comp[idx].set_value(val)
                applied += 1
            except (KeyError, ValueError, AttributeError):
                skipped += 1

    disjunct_cache = {}
    for d in model.component_objects(gdp.Disjunct, descend_into=True):
        disjunct_cache[d.name] = d

    for ind_key, val in snapshot.get("indicators", {}).items():
        try:
            disj_name, idx_part = ind_key.split("::", 1)
        except ValueError:
            skipped += 1
            continue
        disj = disjunct_cache.get(disj_name)
        if disj is None:
            skipped += 1
            continue
        idx = key_to_idx(idx_part)
        try:
            d_obj = disj if idx is None else disj[idx]
            d_obj.binary_indicator_var.set_value(val)
            applied += 1
        except (KeyError, AttributeError):
            skipped += 1

    print(f"Warm start applied: {applied} values set, {skipped} skipped.")
    return applied, skipped


def export_results(model, model_name, filename=None):
    """
    Export optimization results to a structured text file.

    Writes model name, timestamp, objective value, final production,
    unused storage units, and task schedule. The task schedule entries
    are also returned as a list of dicts for downstream plotting.

    Parameters
    ----------
    model : Pyomo ConcreteModel
        A solved Pyomo model with the standard Wine Scheduling variables.
    model_name : str
        Name for the model (printed at the top of the file).
    filename : str, optional
        Output file path. Defaults to "<model_name>_results.txt".

    Returns
    -------
    list[dict]
        Task schedule entries, each containing:
            task  : str
            event : int
            start : float
            end   : float
            batch : float
            units : list[dict] with keys "unit" (str) and "batch" (float)
    """
    if filename is None:
        safe_name = model_name.replace(" ", "_")
        filename = f"{safe_name}_results.txt"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line_product = getattr(model, "line_product", {})
    product_meta = getattr(model, "product_meta", {})

    def split_task(task_name):
        stage = "".join(ch for ch in task_name if ch.isalpha())
        line = "".join(ch for ch in task_name if ch.isdigit())
        return stage, line

    # Collect task schedule data
    def task_active(model, i, n):
        """Return True if task i is active at event n, works for both GDP and non-GDP models."""
        if hasattr(model, "W"):
            return pyo.value(model.W[i, n]) > 0.5
        d_act = model.find_component(f"d_act_{i}_{n}")
        if d_act is not None:
            try:
                val = pyo.value(d_act.binary_indicator_var)
                if val is not None:
                    return val > 0.5
            except Exception:
                pass
        ij_ref = getattr(model, "ij_nonpool", model.ij)
        has_ij = any((i, j) in ij_ref for j in model.j)
        if has_ij:
            return any(
                pyo.value(model.y[i, j, n]) > 0.5 for j in model.j if (i, j) in ij_ref
            )
        return pyo.value(model.b[i, n]) > 1e-3

    task_schedule = []
    for i in model.i:
        for n in model.n:
            if task_active(model, i, n):
                start = pyo.value(model.Ts[i, n])
                end = pyo.value(model.Tf[i, n])
                batch = pyo.value(model.b[i, n])
                units_used = []
                iBAR = set(model.iBAR) if hasattr(model, "iBAR") else set()
                iJAR = set(model.iJAR) if hasattr(model, "iJAR") else set()
                if i in iBAR and hasattr(model, "NumBarr"):
                    num = int(round(pyo.value(model.NumBarr[i, n])))
                    if num > 0:
                        units_used.append(
                            {
                                "unit": f"barriquex{num}",
                                "batch": pyo.value(model.b[i, n]) / num,
                            }
                        )
                elif i in iJAR and hasattr(model, "NumJar"):
                    num = int(round(pyo.value(model.NumJar[i, n])))
                    if num > 0:
                        units_used.append(
                            {
                                "unit": f"jarx{num}",
                                "batch": pyo.value(model.b[i, n]) / num,
                            }
                        )
                else:
                    ij_ref = getattr(model, "ij_nonpool", model.ij)
                    for j in model.j:
                        if (i, j) in ij_ref and pyo.value(model.y[i, j, n]) > 0.5:
                            units_used.append(
                                {
                                    "unit": j,
                                    "batch": pyo.value(model.bj[i, j, n]),
                                }
                            )
                stage, line = split_task(i)
                product_key = line_product.get(line)
                task_schedule.append(
                    {
                        "task": i,
                        "stage": stage,
                        "line": line,
                        "product": product_key,
                        "event": n,
                        "start": start,
                        "end": end,
                        "batch": batch,
                        "units": units_used,
                    }
                )

    # Sort by start time for readability
    task_schedule.sort(key=lambda x: (x["start"], x["task"]))

    # Write output file
    with open(filename, "w", encoding="utf-8") as f:
        sep = "=" * 80

        # Header
        f.write(f"{sep}\n")
        f.write(f"{model_name}\n")
        f.write(f"Generated: {now}\n")
        f.write(f"{sep}\n\n")

        # Objective value
        obj_val = pyo.value(model.OBJ)
        f.write(f"Objective Value: {obj_val:.4f}\n")

        # Economic KPI components (when available)
        if hasattr(model, "Profit"):
            f.write(f"Profit: {pyo.value(model.Profit):.4f}\n")
        if hasattr(model, "Revenue"):
            f.write(f"Revenue: {pyo.value(model.Revenue):.4f}\n")
        if hasattr(model, "GrapeSkinRevenue"):
            f.write(f"Grape Skin Revenue: {pyo.value(model.GrapeSkinRevenue):.4f}\n")
        if hasattr(model, "OutsourcingCost"):
            f.write(f"Outsourcing Cost: {pyo.value(model.OutsourcingCost):.4f}\n")
        if hasattr(model, "RawMaterialCost"):
            f.write(f"Raw Material Cost: {pyo.value(model.RawMaterialCost):.4f}\n")
        if hasattr(model, "LatenessCost"):
            f.write(f"Lateness Cost: {pyo.value(model.LatenessCost):.4f}\n")
        discard_by_product: dict = {}
        if hasattr(model, "DiscardCost"):
            discard_cost = pyo.value(model.DiscardCost)
            if (
                hasattr(model, "Discard")
                and hasattr(model, "SI")
                and hasattr(model, "n")
            ):
                from collections import defaultdict

                _prod_discard: dict = defaultdict(float)
                for s in model.SI:
                    line = "".join(ch for ch in str(s) if ch.isdigit())
                    if line:
                        product_key = line_product.get(line)
                        if product_key:
                            for n in model.n:
                                _prod_discard[product_key] += pyo.value(
                                    model.Discard[s, n]
                                )
                discard_by_product = dict(_prod_discard)
                total_discarded = sum(discard_by_product.values())
            else:
                total_discarded = 0.0
            f.write(
                f"Discard Cost: {discard_cost:.4f} ({total_discarded:.2f} L discarded)\n"
            )
        if hasattr(model, "PenaltyUnused"):
            f.write(f"Penalty Empty Tank: {pyo.value(model.PenaltyUnused):.4f}\n")
        if hasattr(model, "PenaltySpace"):
            f.write(f"Penalty Air Space: {pyo.value(model.PenaltySpace):.4f}\n")
        if hasattr(model, "MakespanPenalty"):
            f.write(f"Penalty Makespan: {pyo.value(model.MakespanPenalty):.4f}\n")
        if hasattr(model, "CoolingCost"):
            f.write(f"Cooling Cost: {pyo.value(model.CoolingCost):.4f}\n")

        # Makespan
        ms = pyo.value(model.MS)
        f.write(f"Makespan: {ms:.2f} hours ({ms / 24:.2f} days)\n")

        # Profit model components
        # if hasattr(model, "Revenue"):
        #     f.write("\nPROFIT COMPONENTS:\n")
        #     f.write(f"  Revenue:              ${pyo.value(model.Revenue):,.2f}\n")
        #     if hasattr(model, "OutsourcingCost"):
        #         f.write(
        #             f"  Outsourcing Cost:    -${pyo.value(model.OutsourcingCost):,.2f}\n"
        #         )
        #     if hasattr(model, "LatenessCost"):
        #         f.write(
        #             f"  Lateness Penalty:    -${pyo.value(model.LatenessCost):,.2f}\n"
        #         )
        #     if hasattr(model, "UnusedTankCost"):
        #         f.write(
        #             f"  Unused Tank Penalty: -${pyo.value(model.UnusedTankCost):,.2f}\n"
        #         )
        #     if hasattr(model, "AirCost"):
        #         f.write(f"  Air Space Penalty:   -${pyo.value(model.AirCost):,.2f}\n")

        # Deadline / lateness (profit model)
        # if hasattr(model, "Deadline"):
        #     deadline = pyo.value(model.Deadline)
        #     f.write(f"\nDeadline: {deadline:.2f} hours ({deadline / 24:.2f} days)\n")
        # if hasattr(model, "Lateness"):
        #     lateness = pyo.value(model.Lateness)
        #     f.write(f"Lateness: {lateness:.2f} hours ({lateness / 24:.2f} days)\n")

        # Unused storage units
        if hasattr(model, "JST_unused"):
            unused = pyo.value(model.JST_unused)
            f.write(f"\nUnused storage units: {unused:.0f}\n")

        # Final production
        f.write(f"\n{sep}\n")
        f.write("FINAL PRODUCTION:\n")
        f.write(f"{'-' * 80}\n")
        outsource_var = None
        if hasattr(model, "Outsource"):
            outsource_var = model.Outsource
        elif hasattr(model, "OutsourcedQty"):
            outsource_var = model.OutsourcedQty

        for s in model.SP:
            prod = pyo.value(model.FinalProd[s])
            demand = pyo.value(model.D[s])
            if demand > 0 or prod > 0.01:
                line = f"{s}: {prod:.2f} L (Demand: {demand:.2f} L)"
                if s in product_meta:
                    category = product_meta[s].get("category", "Unknown")
                    wine_name = product_meta[s].get("name", s)
                    line += f" [{wine_name} | {category}]"
                if outsource_var is not None:
                    try:
                        outsourced = pyo.value(outsource_var[s])
                    except (KeyError, ValueError):
                        outsourced = 0.0
                    if outsourced > 0.01:
                        line += f", Outsourced: {outsourced:.2f} L"
                f.write(line + "\n")

        # Task schedule
        f.write(f"\n{sep}\n")
        f.write("TASK SCHEDULE:\n")
        f.write(f"{sep}\n")
        for entry in task_schedule:
            duration = entry["end"] - entry["start"]
            f.write(f"\nTask {entry['task']}, Event {entry['event']}:\n")
            f.write(f"  Start: {entry['start']:.2f} h, End: {entry['end']:.2f} h\n")
            f.write(f"  Duration: {duration:.2f} h\n")
            f.write(f"  Batch: {entry['batch']:.2f} L\n")
            units_str = "  Units: " + " ".join(
                f"{u['unit']} ({u['batch']:.2f} L)" for u in entry["units"]
            )
            f.write(units_str + "\n")

        # Plotting data
        f.write(f"\n{sep}\n")
        f.write("PLOTTING DATA (task,event,start_h,end_h,batch_L,units):\n")
        f.write(f"{'-' * 80}\n")
        for entry in task_schedule:
            units_str = ";".join(
                f"{u['unit']}:{u['batch']:.2f}" for u in entry["units"]
            )
            f.write(
                f"{entry['task']},{entry['event']},"
                f"{entry['start']:.4f},{entry['end']:.4f},"
                f"{entry['batch']:.4f},"
                f"{units_str}\n"
            )

        # Wine metadata
        if line_product:
            f.write(f"\n{sep}\n")
            f.write("WINE METADATA (line,product,name,category):\n")
            f.write(f"{'-' * 80}\n")
            for line in sorted(line_product.keys(), key=lambda x: int(x)):
                product_key = line_product[line]
                meta = product_meta.get(product_key, {})
                product_name = str(meta.get("name", product_key)).replace(",", " ")
                category = str(meta.get("category", "Unknown")).replace(",", " ")
                f.write(f"{line},{product_key},{product_name},{category}\n")

        # Per-product discard
        if discard_by_product:
            f.write(f"\n{sep}\n")
            f.write("DISCARD DATA (product,discarded_L):\n")
            f.write(f"{'-' * 80}\n")
            for product_key, discarded_vol in sorted(discard_by_product.items()):
                f.write(f"{product_key},{discarded_vol:.4f}\n")

    print(f"Results exported to {filename}")
    return task_schedule
