"""
Utility functions for Wine Scheduling Optimization models.
"""

from datetime import datetime
import pyomo.environ as pyo  # type: ignore


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

    # Collect task schedule data
    def task_active(model, i, n):
        """Return True if task i is active at event n, works for both GDP and non-GDP models."""
        if hasattr(model, "W"):
            return pyo.value(model.W[i, n]) > 0.5
        d_act = model.find_component(f"d_act_{i}_{n}")
        return d_act is not None and pyo.value(d_act.binary_indicator_var) > 0.5

    task_schedule = []
    for i in model.i:
        for n in model.n:
            if task_active(model, i, n):
                start = pyo.value(model.Ts[i, n])
                end = pyo.value(model.Tf[i, n])
                batch = pyo.value(model.b[i, n])
                units_used = []
                for j in model.j:
                    if (i, j) in model.ij and pyo.value(model.y[i, j, n]) > 0.5:
                        units_used.append(
                            {
                                "unit": j,
                                "batch": pyo.value(model.bj[i, j, n]),
                            }
                        )
                task_schedule.append(
                    {
                        "task": i,
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
    with open(filename, "w") as f:
        sep = "=" * 80

        # Header
        f.write(f"{sep}\n")
        f.write(f"{model_name}\n")
        f.write(f"Generated: {now}\n")
        f.write(f"{sep}\n\n")

        # Objective value
        obj_val = pyo.value(model.OBJ)
        f.write(f"Objective Value: {obj_val:.4f}\n")

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
        unused = pyo.value(model.JSTsinusar)
        f.write(f"\nUnused storage units: {unused:.0f}\n")

        # Final production
        f.write(f"\n{sep}\n")
        f.write("FINAL PRODUCTION:\n")
        f.write(f"{'-' * 80}\n")
        for s in model.SP:
            prod = pyo.value(model.ProdFinal[s])
            demand = pyo.value(model.D[s])
            if demand > 0 or prod > 0.01:
                line = f"{s}: {prod:.2f} L (Demand: {demand:.2f} L)"
                if hasattr(model, "OutsourcedQty"):
                    outsourced = pyo.value(model.OutsourcedQty[s])
                    if outsourced > 0.01:
                        line += f", Outsourced: {outsourced:.2f} L"
                f.write(line + "\n")
                if hasattr(model, "FueraDeposito"):
                    out_of_deposit = pyo.value(model.FueraDeposito[s])
                    if out_of_deposit > 0.01:
                        f.write(f"  Out of deposit: {out_of_deposit:.2f} L\n")

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

    print(f"Results exported to {filename}")
    return task_schedule
