# Wine Scheduling

This repository contains the Pyomo implementation and case-study data for
*Exact Label-Free Vessel Pooling for GDP-Based Scheduling of Multi-Campaign
Wine Production*.

The model coordinates production, aging, intermediate holding, outsourcing,
campaign deadlines, and shared equipment over a continuous-time horizon. Its
main formulation contribution is an exact label-free representation of
identical jar and barrique pools.

## Main files

- `WineSchedulingEconomic_GDP.py`: economic scheduling model.
- `utils.py`: result-export and in-memory solution-snapshot utilities.
- `plot_solution.py`: Gantt, production, and state-task-network plotting.
- `parameters.toml`: default model data.
- `scenarios/`: parameter files for the three scenarios in the paper.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.13.11
- Gurobi 13 with a license

The Python dependencies are pinned in `pyproject.toml` and `uv.lock`.

## Environment setup

Run from the repository root:

```powershell
uv sync
```

Gurobi must be installed and licensed separately.

## Running the reported scenarios

The scenario files already contain the event counts used in the manuscript:

- Scenario 1: base case, 11 event points
- Scenario 2: poor harvest, 7 event points
- Scenario 3: demand surge, 13 event points

Run a scenario from the repository root:

```powershell
uv run python WineSchedulingEconomic_GDP.py --params scenarios/parameters_s1.toml
uv run python WineSchedulingEconomic_GDP.py --params scenarios/parameters_s2.toml
uv run python WineSchedulingEconomic_GDP.py --params scenarios/parameters_s3.toml
```

The direct script uses the structure-aware Big-M formulation, a two-hour time
limit, a 3% target MIP gap, and six solver threads.
