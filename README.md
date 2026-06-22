# Wine Scheduling Model

## Table of Contents
- [Wine Scheduling Model](#wine-scheduling-model)
  - [Table of Contents](#table-of-contents)
  - [Makespan minimization](#makespan-minimization)
  - [Sets and Indices](#sets-and-indices)
  - [Parameters](#parameters)
  - [Decision Variables](#decision-variables)
  - [Equations](#equations)
    - [1. Material Balances](#1-material-balances)
    - [2. Task Timing and Sequencing](#2-task-timing-and-sequencing)
    - [3. Unit Assignment and Capacity](#3-unit-assignment-and-capacity)
    - [4. Batch Sizing](#4-batch-sizing)
    - [5. Operational Constraints](#5-operational-constraints)
    - [6. Unit Timing Synchronization](#6-unit-timing-synchronization)
    - [7. Special Constraints](#7-special-constraints)
    - [8. Final State and Objective](#8-final-state-and-objective)
  - [GDP Reformulation](#gdp-reformulation)
    - [1. Combined Task \& Unit Disjunction (Eq. 2.1)](#1-combined-task--unit-disjunction-eq-21)
    - [2. Global Precedence Logic (Eq. 2.2)](#2-global-precedence-logic-eq-22)
    - [3. Storage Persistence via Logical Implications (Eq. 2.3)](#3-storage-persistence-via-logical-implications-eq-23)
    - [4. Algebraic Constraints (Global Coupling)](#4-algebraic-constraints-global-coupling)
    - [5. Variable Definitions (GDP Adaptations)](#5-variable-definitions-gdp-adaptations)
  - [Economic Optimization](#economic-optimization)
    - [Additional Sets](#additional-sets)
    - [Additional Parameters](#additional-parameters)
    - [Additional Variables](#additional-variables)
    - [Economic Constraints](#economic-constraints)
      - [Material Balances (Economic Model)](#material-balances-economic-model)
      - [Demand Constraint](#demand-constraint)
      - [Zero-Inventory Constraints](#zero-inventory-constraints)
      - [Makespan Lower Bound](#makespan-lower-bound)
      - [Ecobulk Stg Maximum Duration](#ecobulk-stg-maximum-duration)
      - [Two-Path Intermediate Stg Storage](#two-path-intermediate-stg-storage)
      - [Static Preprocessing (Event-Range Fixing)](#static-preprocessing-event-range-fixing)
      - [Batch and Duration Linking](#batch-and-duration-linking)
      - [Identical-Unit Symmetry Breaking](#identical-unit-symmetry-breaking)
      - [Remaining Economic Constraints](#remaining-economic-constraints)
    - [Economic Objective](#economic-objective)
    - [Barrique and Jar Pool Aggregation](#barrique-and-jar-pool-aggregation)

## Makespan minimization

This is a continuous-time, event-based scheduling model (STN-style) for wine production (`WineSchedulingMS.py`).
Time is discretized into event points, not fixed intervals. At each event, tasks may start, finish, produce material, or consume material.

The model decides:

- which tasks run,
- in which units,
- at which event points,
- with what batch sizes,

while respecting material balances, timing, unit capacity, storage rules, and wine-specific constraints (zero-wait, no-intermediate-storage, Ecobulk).

The objective minimizes the makespan while steering storage usage.

## Sets and Indices

- $I$: Set of tasks (index $i$) (operations like fermentation, transfer, aging, blending, storage)
- $J$: Set of units (index $j$) (tanks, reactors, storage vessels)
- $N$: Set of event points (index $n$) (impose an order: event n happens before n+1)
- $S$: Set of states (index $s$) (materials: must, wine, intermediates, final products)
  
Special subsets encode operational rules:

- $I^{\mathrm{st}} \subset I$: Storage tasks
- $I^{\mathrm{nst}} \subset I$: Non-storage tasks
- $I^{\mathrm{pst}} \subset I$: Pre-storage tasks
- $J^{\mathrm{st}} \subset J$: Storage units
- $S^{\mathrm{P}} \subset S$: Final products
- $S^{\mathrm{ZW}} \subset S$: Zero-wait states (includes shared must pool $m$)
- $S^{\mathrm{NIS}} \subset S$: No-intermediate-storage states
- $\mathrm{IJ} \subseteq I \times J$: Feasible task-unit assignment pairs
- $J_i = \{j : (i,j) \in \mathrm{IJ}\}$: Units compatible with task $i$
- $I_j = \{i : (i,j) \in \mathrm{IJ}\}$: Tasks compatible with unit $j$

This lets the same formulation handle very different behaviors just by set membership.

## Parameters

- $\alpha_i$: Fixed processing time for task $i$
- $\beta_i$: Variable processing time coefficient for task $i$
- $\rho_{i,s}^{\mathrm{prod}}$: Production coefficient of state $s$ by task $i$
- $\rho_{i,s}^{\mathrm{cons}}$: Consumption coefficient of state $s$ by task $i$
- $B_{i,j}^{\mathrm{min}}, B_{i,j}^{\mathrm{max}}$: Min/Max batch size for task $i$ in unit $j$
- $H$: Time horizon
- $D_s$: Demand for state $s$

Activation and unit-assignment limits:

- $i_{\mathrm{max}}$: Default maximum number of activations for non-storage tasks (typically 1)
- $\mathrm{task}_{i_{\mathrm{max},i}}$: Per-task activation limit. Defaults to $i_{\mathrm{max}}$, overridden for the shared pressing task to equal the number of pressing lines
- $j_{\mathrm{max}}$: Maximum number of units assigned to an active non-storage task at an event (overridden per task via `max_units` in the template)
- $j_{\mathrm{min}}$: Minimum number of units assigned to an active non-storage task at an event
- $i_{\mathrm{max,ST}} \in \{0,1\}$: Binary flag enabling (1) or forbidding (0) storage tasks (applied at the final event for the makespan model, storage tasks persist, so they activate at most once)
- $j_{\mathrm{max,ST}}$: Maximum number of units assigned to an active storage task at an event
- $j_{\mathrm{min,ST}}$: Minimum number of units assigned to an active storage task at an event

## Decision Variables

- $W_{i,n} \in \{0,1\}$: 1 if task $i$ is active in event $n$ (does task i occur at event n?)
- $y_{i,j,n} \in \{0,1\}$: 1 if task $i$ uses unit $j$ in event $n$ (if it occurs, which unit does it use?)
- $b_{i,n} \ge 0$: Batch size of task $i$ in event $n$ (total amount of wine processed in that event?)
- $b_{i,j,n} \ge 0$: Batch size of task $i$ in unit $j$ in event $n$ (how much wine is processed in that unit?)
- $\mathrm{ST}_{s,n} \ge 0$: Inventory of state $s$ at event $n$ (how much material exists after each event)
- $\mathrm{Ts}_{i,n}, \mathrm{Tf}_{i,n} \ge 0$: Start and finish times of task $i$ in event $n$
- $\mathrm{Tsj}_{j,n}, \mathrm{Tfj}_{j,n} \ge 0$: Start and finish times of unit $j$ in event $n$
- $\mathrm{MS}$: Makespan (total schedule length)

## Equations

### 1. Material Balances

**Initial Event ($n=1$) (h04):**

$$ \mathrm{ST}_{s,1} = \mathrm{ST}_{s,0} + \sum_{i \in \mathrm{ICS}_s} \rho_{i,s}^{\mathrm{cons}} b_{i,1} $$

Inventory starts from an initial stock and subtracts what is consumed at event 1.

**Subsequent Events ($n > 1$) (h03):**

$$ \mathrm{ST}_{s,n} = \mathrm{ST}_{s,n-1} + \sum_{i \in \mathrm{IPS}_s} \rho_{i,s}^{\mathrm{prod}} b_{i,n-1} + \sum_{i \in \mathrm{ICS}_s} \rho_{i,s}^{\mathrm{cons}} b_{i,n} $$

Inventory evolves as:

- previous inventory
- plus production from tasks at the previous event
- minus consumption from tasks at the current event

### 2. Task Timing and Sequencing

**Processing Time (h05):**
For non-storage tasks ($i \in I^{\mathrm{nst}}$):

$$ \mathrm{Tf}_{i,n} = \mathrm{Ts}_{i,n} + \alpha_i W_{i,n} + \beta_i b_{i,n} $$

- `duration = fixed time + batch-dependent time`
- multiplied by activation `W`
- If the task is inactive, duration is equal to zero.

**Task Sequence (g06):**

$$ \mathrm{Ts}_{i,n+1} \ge \mathrm{Tf}_{i,n} $$

A task at event `n+1` cannot start before it finishes at `n`.
This creates a natural temporal chain across events.

**Task Precedence (g11):**
If task $i$ consumes what task $i'$ produces:

$$ \mathrm{Ts}_{i,n+1} \ge \mathrm{Tf}_{i',n} - H(1 - W_{i',n}) $$

- `i` at `n+1` must start after `i′` at `n` finishes
- Big-M deactivates the constraint when tasks are inactive

**Zero Wait (ZW) Constraint (g12):**
For states $s \in S^{\mathrm{ZW}}$, if $i$ consumes $s$ from $i'$:

$$ \mathrm{Ts}_{i,n+1} \le \mathrm{Tf}_{i',n} + H(2 - W_{i',n} - W_{i,n+1}) $$

Zero-wait: consumption must start exactly when production ends

**No Intermediate Storage (NIS) Constraint (g13):**
For states $s \in S^{\mathrm{NIS}}$, if $i$ consumes $s$ from $i'$:

$$ \mathrm{Ts}_{i,n+1} \le \mathrm{Tf}_{i',n} + H(2 - W_{i',n} - W_{i,n+1}) $$

No-intermediate-storage: material cannot sit in inventory

### 3. Unit Assignment and Capacity

Total time spent by all non-storage tasks on a unit cannot exceed the schedule length (makespan). This is a load/capacity bound, ordering is handled by the unit sequencing constraints (A09/A10).

Each task:

- needs at least $j_{\mathrm{min}}$ units
- can use at most $j_{\mathrm{max}}$ units
- only if the task is active

This supports parallel units (e.g. multiple tanks for one blend).

**Unit Capacity / Load (h09):**

$$ \sum_{i \in I^{\mathrm{nst}}} \sum_{n} (\alpha_i y_{i,j,n} + \beta_i b_{i,j,n}) \le MS $$

**Max Units per Task (A01, A01st):**

$$ \sum_{j \in J_i} y_{i,j,n} \le j_{\mathrm{max}} \cdot W_{i,n} $$

**Min Units per Task (A02, A02st):**

$$ \sum_{j \in J_i} y_{i,j,n} \ge j_{\mathrm{min}} \cdot W_{i,n} $$

**Single Task per Unit (A03):**

$$ \sum_{i \in I_j} y_{i,j,n} \le 1 $$

Each unit can handle at most one task per event.

**Shared pressing task.** All pressing lines (white and rosé) share a single `Pr` task and a single zero-wait must pool state $m$. The `Pr` task consumes grapes from the shared raw-material pool and produces into $m$; each fermentation task for a pressed wine consumes from $m$. This eliminates artificial per-line symmetry: the press can be scheduled once and serve multiple fermentation tasks at the next event. The per-task activation limit is tightened to

$$\mathrm{task}_{i_{\mathrm{max},\mathrm{Pr}}} = \min\!\bigl(n_{\mathrm{press}},\ n_{\mathrm{max}} - (K_{\mathrm{min}} - 1)\bigr)$$

where $n_{\mathrm{press}}$ is the number of pressing lines and $K_{\mathrm{min}}$ is the step count of the shortest pressing line. The second term caps `Pr` by the number of events that can precede the final downstream step: a press run at event $n$ requires at least $K_{\mathrm{min}}-1$ more events after it, so `Pr` can feasibly start at most $n_{\mathrm{max}}-(K_{\mathrm{min}}-1)$ times. In the makespan models (where outsourcing is unavailable) a complementary lower bound is also imposed.

### 4. Batch Sizing

**Batch Aggregation (A04):**

$$ b_{i,n} = \sum_{j \in J_i} b_{i,j,n} $$

Total batch size equals the sum over assigned units.

**Batch Limits (A05a, A05b):**

$$ B_{i,j}^{\mathrm{min}} y_{i,j,n} \le b_{i,j,n} \le B_{i,j}^{\mathrm{max}} y_{i,j,n} $$

If a unit is used, batch size must lie between min and max.
If it is not used, batch size is forced to zero.

**Fixed-capacity tightening.** When $B_{i,j}^{\mathrm{min}} = B_{i,j}^{\mathrm{max}} = B$ (most subterranean tanks, barriques, jars, ecobulks), the two inequalities collapse to one equality inside the assign disjunct:

$$ b_{i,j,n} = B \cdot y_{i,j,n} $$

### 5. Operational Constraints

**Max Activations - Non-storage tasks (A06):**

$$ \sum_{n} W_{i,n} \le \mathrm{task}_{i_{\mathrm{max},i}} \quad \forall i \in I^{\mathrm{nst}} $$

Limits how many times a non-storage task can occur. For most tasks $\mathrm{task}_{i_{\mathrm{max},i}} = i_{\mathrm{max}} = 1$ (one fermentation per batch). The shared pressing task uses the tighter bound $\mathrm{task}_{i_{\mathrm{max},\mathrm{Pr}}} = \min(n_{\mathrm{press}},\ n_{\mathrm{max}}-(K_{\mathrm{min}}-1))$ as described above.

**Min Activations of Pr - Makespan models (A06_min_Pr):**

In the makespan models all demand must be met in-house (no outsourcing), so the total must produced by the press must equal at least the sum of each pressing line's demand divided by its chain yield from must to final product:

$$ \sum_{n} W_{\mathrm{Pr},n} \ge \left\lceil \frac{\displaystyle\sum_{l \in L_{\mathrm{press}}} D_{s_l}\,/\,\eta_l}{\max_j B^{\mathrm{max}}_{\mathrm{Pr},j} \cdot \rho^{\mathrm{prod}}_{\mathrm{Pr},m}} \right\rceil $$

where $\eta_l = \prod_k \rho^{\mathrm{prod}}_k$ is the chain yield from must to final product along line $l$ and $\rho^{\mathrm{prod}}_{\mathrm{Pr},m}$ is the must yield of the press.

**Storage enable - Storage tasks (A06st):**

$$ W_{i,N} \le i_{\mathrm{max,ST}} \quad \forall i \in I^{\mathrm{st}} $$

With the persistence chain (A14, see below), $W_{i,n}$ is non-decreasing, so $W_{i,N}$ is 0/1 = whether the task ever started. This is why $i_{\mathrm{max,ST}} \in \{0,1\}$ acts as a global switch: storage permitted when 1 (constraint trivial), forbidden when 0 (forces $W_{i,N}=0$).

**Storage Task Duration (A07, A08):**

$$ \mathrm{Tf}_{i,n} \ge \mathrm{Ts}_{i,n} $$

$$ \mathrm{Tf}_{i,n} \ge H \cdot W_{i,n} $$

Storage tasks:

- cannot have negative duration
- once activated, effectively occupy the unit for the whole horizon

This models tanks that stay filled once used.

### 6. Unit Timing Synchronization

These constraints lock unit timelines to task timelines.

If task `i` uses unit `j` at event `n`, then:

- unit start time equals task start time
- unit finish time equals task finish time

Otherwise, constraints are relaxed via Big-M.

This prevents subtle inconsistencies like a unit being busy when the task is not.

**Unit Sequence (A09):**

$$ \mathrm{Tsj}_{j,n+1} \ge \mathrm{Tfj}_{j,n} $$

**Unit Duration (A10):**

$$ \mathrm{Tfj}_{j,n} \ge \mathrm{Tsj}_{j,n} $$

**Sync Start Times (A11a, A11b):**

$$ \mathrm{Tsj}_{j,n} \le \mathrm{Ts}_{i,n} + H(1 - y_{i,j,n}) $$

$$ \mathrm{Tsj}_{j,n} \ge \mathrm{Ts}_{i,n} - H(1 - y_{i,j,n}) $$

**Sync End Times (A12a, A12b):**

$$ \mathrm{Tfj}_{j,n} \le \mathrm{Tf}_{i,n} + H(1 - y_{i,j,n}) $$

$$ \mathrm{Tfj}_{j,n} \ge \mathrm{Tf}_{i,n} - H(1 - y_{i,j,n}) $$

### 7. Special Constraints

**One-to-One Pairs (A13a):**

$$ W_{i',n+1} = W_{i,n} $$

For tightly coupled tasks (e.g. transfer immediately after fermentation), activation is forced to match across events.

**Storage Persistence (A14, A15):**
If storage task $i$ is active in $n$, it must remain active in $n+1$ (and by transitivity, all future events):

$$ W_{i,n+1} \ge W_{i,n} \quad \forall n \in \{1 \dots N-1\}, i \in I^{\mathrm{st}} $$

$$ y_{i,j,n} \implies y_{i,j,n+1} \quad \forall n \in \{1 \dots N-1\}, (i,j) \in \mathrm{IJ}, i \in I^{\mathrm{st}} $$

Once a storage task starts:

- it must remain active in all future events
- same for unit assignment

This prevents emptying and reusing storage implicitly. The recursive chain is equivalent to the all-pairs formulation but uses $O(N)$ constraints instead of $O(N^2)$.

**Ecobulk Timing (A16):**
For states $s \in S^{\mathrm{FISEco}}$ (Ecobulk compatible), if task $i$ consumes $s$ produced by task $i'$:

$$ \mathrm{Ts}_{i,n+1} \le \mathrm{Tf}_{i',n} + M(2 - W_{i',n} - W_{i,n+1}) $$

This enforces a "no intermediate storage" condition for Ecobulk products. The consuming task must start immediately after the producing task finishes (or before, effectively zero-wait when combined with precedence).

### 8. Final State and Objective

**Demand Satisfaction (g17):**

$$ \mathrm{FinalProd}_s \ge D_s $$

Final production must meet or exceed demand.

**Final Production Calculation (A18):**

$$ \mathrm{ST}_{s,N} + \sum_{i \in \mathrm{IPS}_s} \rho_{i,s}^{\mathrm{prod}} b_{i,N} = \mathrm{FinalProd}_s $$

Final inventory plus last-event production equals total produced.

**Makespan (A21):**

$$ \mathrm{Tf}_{i,N} \le \mathrm{MS} \quad \forall i \in I^{\mathrm{pst}} $$

The finish time of all pre-storage tasks must be ≤ MS.

**Critical-path lower bound (`ms_lb`):** the makespan is bounded below by the longest production line's total fixed processing time:

$$ \mathrm{MS} \ge \max_{l \in L} \sum_{i \in l} \alpha_i $$

**Objective Function:**

$$ \min Z = \mathrm{MS} + \mathrm{Penalty}_{\mathrm{unused}} + \mathrm{Penalty}_{\mathrm{space}} $$

Unused-storage term:

$$ \mathrm{JST}_{\mathrm{unused}} = n_{\mathrm{JST}} - \sum_{j \in \mathrm{JST}}\sum_{\substack{i \in I^{\mathrm{st}}:\\(i,j)\in \mathrm{IJ}}} y_{i,j,N} $$

With storage persistence, this counts tanks that are inactive at the final event and is equivalent to tanks never used.

$$ \mathrm{Penalty}_{\mathrm{unused}} = c^{\mathrm{empty}} \cdot \frac{1}{n_{\mathrm{JST}}} \cdot \mathrm{JST}_{\mathrm{unused}} $$

$$ \mathrm{Penalty}_{\mathrm{space}} = c^{\mathrm{air}} \cdot \frac{1}{\sum_{j\in \mathrm{JST}}(\Delta B)/n_{\mathrm{JST}}} \cdot \sum_{j\in \mathrm{JST}} \mathrm{Freespace}_j $$

Minimize makespan while trying to:

- penalize unused capacity
- reward efficient storage usage

## GDP Reformulation

The reformulation in this model (`WineSchedulingMS_GDP.py`) uses Disjunctive Programming to capture logical `OR` decisions (Run vs. Don't Run, Assign vs. Idle) directly, avoiding manual Big-M constraints for core operational logic.

### 1. Combined Task & Unit Disjunction (Eq. 2.1)

Task activation is an outer disjunction: when a task is active, a separate nested disjunction for each compatible unit $j \in J_i$ decides whether that unit is assigned (`assign` vs. `skip`). If the task is inactive, everything is zeroed out.

$$\forall i \in I, n \in N:
\begin{bmatrix}
W_{i,n} \\
\mathrm{Tf}_{i,n} = \mathrm{Ts}_{i,n} + \alpha_i + \beta_i b_{i,n} \quad (i \in I^{\mathrm{nst}}) \\
\forall j \in J_i:
\begin{bmatrix}
y_{i,j,n} = 1 \\
B^{\mathrm{min}}_{i,j} \le b_{i,j,n} \le B^{\mathrm{max}}_{i,j} \\
\mathrm{Tsj}_{j,n} = \mathrm{Ts}_{i,n} \\
\mathrm{Tfj}_{j,n} = \mathrm{Tf}_{i,n}
\end{bmatrix}
\underline{\vee}
\begin{bmatrix}
y_{i,j,n} = 0 \\
b_{i,j,n} = 0
\end{bmatrix}
\end{bmatrix}
\underline{\vee}
\begin{bmatrix}
\neg W_{i,n} \\
b_{i,n} = 0 \\
\mathrm{Tf}_{i,n} = \mathrm{Ts}_{i,n} \\
\forall j \in J_i:
\begin{bmatrix}
y_{i,j,n} = 0 \\
b_{i,j,n} = 0
\end{bmatrix}
\end{bmatrix}$$

A global constraint enforces at most one task per unit per event:

$$ \sum_{i \in I_j} y_{i,j,n} \le 1 \quad \forall j, n $$

**Symmetry breaking (inactive disjunct).** When a task is inactive at event $n > 1$, its start time is collapsed to the finish of the previous event:

$$\neg W_{i,n} \implies \mathrm{Ts}_{i,n} = \mathrm{Tf}_{i,n-1} \quad \forall i \in I,\ n > 1$$

This prevents $\mathrm{Ts}_{i,n}$ from floating freely in $[0, H]$ when the task is idle, which otherwise creates flat LP plateaus and symmetric integer solutions. (In the economic model this collapse is moved out of the disjunct into a global tight-$M$ constraint (see [Batch and Duration Linking](#batch-and-duration-linking).)

**Variable time-window bounds.** In addition to using $\mathrm{ES}_i$ and $\mathrm{LF}_i$ in Big-M computation, they are applied as direct variable bounds before solving:

$$\mathrm{Ts}_{i,n} \ge \mathrm{ES}_i, \quad \mathrm{Tf}_{i,n} \le \mathrm{LF}_i, \quad \mathrm{Ts}_{i,n} \le \mathrm{LF}_i$$

### 2. Global Precedence Logic (Eq. 2.2)

Instead of 3-way disjunctions, precedence is expressed as conditional constraints using task-pair-specific Big-M implications on $W$.

For each producer-consumer pair $(i', i)$ linked through a shared state, define earliest start and latest finish times from minimum cumulative step durations:

$$\mathrm{ES}_i = \sum_{k < \text{pos}(i)} \alpha_k, \qquad \mathrm{LF}_i = H - \sum_{k > \text{pos}(i)} \alpha_k$$

These yield tight, pair-specific Big-M values:

$$M^\ge_{i',i} = \max(0,\ \mathrm{LF}_{i'} - \mathrm{ES}_i), \qquad M^\le_{i',i} = \max(0,\ \mathrm{LF}_i - \mathrm{ES}_{i'})$$

**General Precedence (`prec_ge`):**

$$ W_{i',n} \land W_{i,n+1} \implies \mathrm{Ts}_{i,n+1} \ge \mathrm{Tf}_{i',n} $$

Linearized as:

$$ \mathrm{Ts}_{i,n+1} \ge \mathrm{Tf}_{i',n} - M^\ge_{i',i}(2 - W_{i',n} - W_{i,n+1}) $$

**Zero-Wait / No-Intermediate-Storage / Ecobulk (equality when both active):**

$$ W_{i',n} \land W_{i,n+1} \implies \mathrm{Ts}_{i,n+1} = \mathrm{Tf}_{i',n} $$

Linearized as the $\ge$ above plus:

$$ \mathrm{Ts}_{i,n+1} \le \mathrm{Tf}_{i',n} + M^\le_{i',i}(2 - W_{i',n} - W_{i,n+1}) $$

Using $M_{i',i} \ll H$ for long-aging lines (up to 85% tighter than the global horizon) significantly strengthens the LP relaxation at the root node.

### 3. Storage Persistence via Logical Implications (Eq. 2.3)

Persistence rules for storage tasks are expressed as pure logical implications on disjunct indicator variables, rather than algebraic inequality chains:

$$ W_{i,n} \implies W_{i,n+1} \quad \forall i \in I^{\mathrm{st}}, n < N $$

This means once a storage task activates at event $n$, it must remain active for all subsequent events. The unit-level persistence ($y_{i,j,n} \implies y_{i,j,n+1}$) is retained as an algebraic constraint (A15).

### 4. Algebraic Constraints (Global Coupling)

These constraints remain algebraic as they involve aggregation across the entire horizon or multiple units.

**Material Balances:**

$$ \mathrm{ST}_{s,n} = \mathrm{ST}_{s,n-1} + \sum \rho^{\mathrm{prod}} b_{n-1} + \sum \rho^{\mathrm{cons}} b_{n} $$

**Batch Aggregation (A04):**

$$ b_{i,n} = \sum_{j \in J_i} b_{i,j,n} $$

**Global Capacity (A06 / A06_min_Pr):**

$$ \sum_{n} W_{i,n} \le \mathrm{task}\_i_{\mathrm{max},i} $$

For the shared pressing task, $\mathrm{task}_{i_{\mathrm{max},\mathrm{Pr}}}$ is bounded above by $\min(n_{\mathrm{press}}, n_{\mathrm{max}}-(K_{\mathrm{min}}-1))$ and, in the makespan models, below by $\lceil \mathrm{TotalMust} / \mathrm{MaxMustPerRun} \rceil$.

**Unit Constraints:**

$$ \sum_{j \in J_i} y_{i,j,n} \le j_{\mathrm{max}} \cdot W_{i,n} $$

**Cover cuts (LP tightening).** After the disjunction structure is built, the following valid inequality is added for every non-pool task $(i, n)$:

$$b_{i,n} \le \sum_{j \in J_i} B^{\mathrm{max}}_{i,j} \cdot y_{i,j,n}$$

This directly links the continuous batch variable to the binary unit-assignment variables without relying on the solver to derive it from the Big-M expansion of the disjunctive bounds.

### 5. Variable Definitions (GDP Adaptations)

- **$W_{i,n}$:** The GDP disjunct's `binary_indicator_var`, no separate binary variable is declared. This single binary serves both the disjunction structure and algebraic constraints (A01, A06, precedence, persistence, etc.), and activates terminal storage horizon sync inside the final storage disjunct.
- **$y_{i,j,n}$:** Unit assignment binary; retained for global constraints (A03, batch limits).

## Economic Optimization

This section documents the economic variant implemented in `WineSchedulingEconomic_GDP.py` using `parameters.toml`. The scheduling structure is the same GDP formulation described above, with added market variables, raw-material costs, aging logic, and a profit-maximizing objective.

### Additional Sets

- $S^{\mathrm{Market}} \subset S^{\mathrm{P}}$: Final product states sold on the market
- $S^{\mathrm{R}} \subset S$: Shared raw-material pool states (e.g. $s_{\mathrm{red}}$, $s_{\mathrm{white\_rose}}$)
- $S^{\mathrm{I}} \subset S$: Intermediate states (all states that are neither final products nor raw pools). The `Discard` variable is defined on this set.
- $S^{\mathrm{Age}} \subset S$: Post-aging, pre-cold-stabilization intermediate states ($\mathrm{va}_l$ for lines where aging precedes cold stabilization). Treated as NIS - cold stabilization must begin immediately after aging finishes.
- $I^{\mathrm{Age}} \subseteq I$: Aging tasks ($\mathrm{AgeBar}^*$, $\mathrm{AgeJar}^*$). Each product $s$ has at most one aging task $i^{\mathrm{Age}}_s$.
- $J^{\mathrm{ext}} \subseteq J$: Exterior units that require active cooling (flagged via `exterior = true` in the TOML).
- $L^{\mathrm{stg}} \subseteq L$: Production lines that include an intermediate Stg (storage holding) step before aging.
- $I^{\mathrm{stg}} \subseteq I$: Stg holding tasks, one per line in $L^{\mathrm{stg}}$ (e.g. $\mathrm{Stg}_l$).
- $I^{\mathrm{stgVar}} \subseteq I$: Stg-path aging variant tasks, one per line in $L^{\mathrm{stg}}$ (e.g. $\mathrm{AgeBar4MStg}_l$). These consume the buffer state $\mathrm{vbuf}_l$ instead of $\mathrm{vl}_l$.
- $S^{\mathrm{vbuf}} \subset S^{\mathrm{ZW}}$: Intermediate buffer states, one per stg line ($\mathrm{vbuf}_l$). Classified as zero-wait.

### Additional Parameters

- $\mathrm{Price}_s$: selling price of product state $s$
- $C^{\mathrm{out}}_s$: outsourcing cost of product state $s$
- $\mathrm{Deadline}_k$: ordered campaign deadlines for young wine lines (no aging). Campaign $k$ must finish by $D_k$.
- $\mathrm{AgingHours}_s$: fixed processing duration of the aging step for product $s$ (0 if the product has no aging step)
- $c^{\mathrm{late}}$: lateness penalty coefficient (cost per hour per product)
- $c^{\mathrm{MS}}$: makespan penalty coefficient (cost per hour of schedule length)
- $c^{\mathrm{cool}}$: cooling cost per liter-hour for exterior tanks ($/(L \cdot h)$)
- $C_s^{\mathrm{raw}}$: raw-material cost per liter for shared raw pools $s \in S^{\mathrm{R}}$
- $c^{\mathrm{discard}}$: cost per liter discarded from intermediate states
- $i_{\mathrm{max}}$ (`iMax`): global maximum activations for non-young aging tasks. Defaults to 1 (one campaign per aged wine line). Setting $i_{\mathrm{max}} = 2$ allows each aged wine line to run two full batches within the horizon (e.g. two consecutive vintages of barrelled wine). All tasks on aged wine lines inherit this limit unless overridden.
- $i_{\mathrm{max,young}}$ (`iMaxYoungWine`): number of campaigns for young wine lines. When $> 1$, multiple separate batches of young wines are scheduled within one horizon of aged wines. Demand and upper bounds in the TOML are specified per campaign. The model multiplies them by $i_{\mathrm{max,young}}$ internally. Affects the `Pr` activation ceiling and all steps of young wine lines:

$$\mathrm{task}_{i_{\mathrm{max},\mathrm{Pr}}} = \min\bigl(n_{\mathrm{young}}^{\mathrm{press}} \cdot i_{\mathrm{max,young}} + n_{\mathrm{aged}}^{\mathrm{press}} \cdot i_{\mathrm{max}},\ n_{\mathrm{max}} - (K_{\mathrm{min}} - 1)\bigr)$$

$$\mathrm{task}_{i_{\mathrm{max},i}} = i_{\mathrm{max,young}} \quad \forall i \in \text{steps of young wine lines} \setminus \{\mathrm{Pr}\}$$

where $n_{\mathrm{young}}^{\mathrm{press}}$ is the number of pressing lines whose product has no aging stage and $n_{\mathrm{aged}}^{\mathrm{press}}$ is the remaining pressing lines. Each aged pressing line may need up to $i_{\mathrm{max}}$ press runs, hence the $i_{\mathrm{max}}$ multiplier.

**Cs task_imax for stg lines.** For lines $l \in L^{\mathrm{stg}}$, cold stabilization ($\mathrm{Cs}_l$) must follow every aging activation regardless of which path was taken. Its per-task limit is set dynamically:

$$\mathrm{task}_{i_{\mathrm{max},\mathrm{Cs}_l}} = \mathrm{task}_{i_{\mathrm{max},\mathrm{AgeDirect}_l}} + \mathrm{task}_{i_{\mathrm{max},\mathrm{AgeViaStg}_l}} \quad \forall l \in L^{\mathrm{stg}}$$

This ensures the solver is never forced to discard post-aging wine simply because the $\mathrm{Cs}$ limit was tighter than the combined aging throughput.

### Additional Variables

- $\mathrm{Outsource}_s \ge 0$: outsourced quantity of product $s$
- $\mathrm{Late}_{s,k} \ge 0$: delay of product $s$ beyond deadline $D_k$ for campaign $k \in \{1,\ldots,K\}$. Defined for all non-aging products at $k=1$ and for young wine products (no aging) at $k>1$.
- $\mathrm{MS} \ge 0$: makespan (retained from scheduling model)
- $\mathrm{FinalProd}_s \ge 0$: final internally produced quantity of product state $s$
- $\mathrm{Discard}_{s,n} \ge 0$: volume (L) discarded from intermediate state $s \in S^{\mathrm{I}}$ at event $n$. Allows the consuming task's batch to be smaller than the upstream produced quantity when a fixed-capacity downstream vessel cannot accept the full volume.

### Economic Constraints

#### Material Balances (Economic Model)

The standard material balance (h03/h04) is extended with a `Discard` term on all intermediate states ($s \in S^{\mathrm{I}}$):

**Initial event ($n=1$):**

$$ \mathrm{ST}_{s,1} = \mathrm{ST}_{s,0} + \sum_{i \in \mathrm{ICS}_s} \rho_{i,s}^{\mathrm{cons}} b_{i,1} - \mathrm{Discard}_{s,1} \quad \forall s \in S^{\mathrm{I}} $$

**Subsequent events ($n > 1$):**

$$ \mathrm{ST}_{s,n} = \mathrm{ST}_{s,n-1} + \sum_{i \in \mathrm{IPS}_s} \rho_{i,s}^{\mathrm{prod}} b_{i,n-1} + \sum_{i \in \mathrm{ICS}_s} \rho_{i,s}^{\mathrm{cons}} b_{i,n} - \mathrm{Discard}_{s,n} \quad \forall s \in S^{\mathrm{I}} $$

For raw-material pools ($s \in S^{\mathrm{R}}$) and final product states ($s \in S^{\mathrm{P}}$), the original balances without `Discard` are retained.

**Effect on zero-wait / NIS states.** For states that enforce zero inventory ($\mathrm{ST}_{s,n}=0$ always), the balance collapses to:

$$ \sum_{i \in \mathrm{ICS}_s} \rho_{i,s}^{\mathrm{cons}} b_{i,n} = \sum_{i \in \mathrm{IPS}_s} \rho_{i,s}^{\mathrm{prod}} b_{i,n-1} - \mathrm{Discard}_{s,n} $$

Without `Discard`, the consuming batch is fully determined by the producing batch and the yield coefficients. With `Discard`, the constraint becomes $\le$. The consuming batch may be any fraction of the upstream volume and the excess is discarded. This allows the model to match fixed-capacity downstream vessels to variable upstream batches without infeasibility.

**Terminal balance (h16).** Zero-wait states require that any production at the final event $N$ (which has no successor event for a consuming task) is also explicitly accounted for. The constraint is extended to allow discarding this stranded volume:

$$ \mathrm{ST}_{s,N} + \sum_{i \in \mathrm{IPS}_s} \rho_{i,s}^{\mathrm{prod}} b_{i,N} - \mathrm{Discard}_{s,N} = 0 \quad \forall s \in S^{\mathrm{ZW}} $$

NIS states do not need an analogous terminal constraint: the NIS condition (`g13`) prevents any task producing into an NIS state from running at $N$ (it would require a consuming task at $N+1$ which does not exist), so production at the final event is always zero for these states.

#### Demand Constraint

Demand can be met by internal production plus outsourcing. For products with `OutsourceAllowed = 0`, the outsourcing term is blocked, so demand must be met entirely in-house:

$$ \mathrm{FinalProd}_s + \mathrm{Outsource}_s \ge D_s \quad \forall s \in S^{\mathrm{Market}} $$

**Minimum-activation valid cuts (A06_min).** For each non-outsourceable line, the cut is applied to all non-press tasks in the line (not only the last task). For each such task $i$, at least $\lceil D_{s_l} / \bar{B}_i \rceil$ activations are required, where $\bar{B}_i$ is the maximum batch size achievable in a single activation:

- For non-pool tasks: $\bar{B}_i = \max_{j:(i,j)\in \mathrm{IJ}} B^{\mathrm{max}}_{i,j}$ (largest individual unit).
- For pool tasks (barrique/jar): $\bar{B}_i = \bar{n}_{\mathrm{pool}} \cdot B_{\mathrm{pool}}$ (entire pool in one activation).

$$\sum_{n \in N} W_{i,n} \ge \left\lceil \frac{D_{s_l}}{\bar{B}_i} \right\rceil \quad \forall l : s_l \notin \mathrm{Outsource},\ \forall i \in \mathrm{chain}(l)\setminus\{\mathrm{Pr}\}$$

Using the per-unit capacity $B^{\mathrm{max}}_{i,j}$ for pool tasks instead of $\bar{n}_{\mathrm{pool}}\cdot B_{\mathrm{pool}}$ would overstate the required activations (e.g. $\lceil 5500/250 \rceil = 22$ barriques required versus the correct $\lceil 5500/21250 \rceil = 1$). With $i_{\mathrm{max}}=2$ this error clamps to $\min(22,2)=2$, imposing two runs of a 15-month aging step in a 16000 h horizon, an infeasible constraint. The cut is only added when the result is $\ge 2$, so with correct pool capacities it usually fires only for Stg or fermentation tasks where demand genuinely exceeds a single-tank fill.

**Shared pressing minimum-activation cut (A06_min_Pr).** In the economic model this cut is now added conditionally for non-outsource pressed lines, using the required total must volume and press yield (same structure as the makespan version, with outsourcing-aware filtering).

#### Zero-Inventory Constraints

Zero-wait and NIS states have $\mathrm{ST}_{s,n} = 0$ enforced explicitly at every event as an algebraic constraint, not only through the material balance structure:

$$\mathrm{ST}_{s,n} = 0 \quad \forall s \in S^{\mathrm{ZW}} \cup S^{\mathrm{NIS}},\ n \in N$$

These are added directly so the solver sees the equality without having to infer it from the interaction of h03/h04, the ZW/NIS precedence constraints, and the demand constraint.

#### Makespan Lower Bound

A valid lower bound on the makespan is derived from the minimum total processing time along the longest production line:

$$\mathrm{MS} \ge \max_{l \in L} \sum_{k \in \text{steps}(l)} \alpha_k$$

#### Ecobulk Stg Maximum Duration

Ecobulk tanks must not hold wine for more than 720 hours (1 month). For each Ecobulk-compatible Stg task $i \in I^{\mathrm{stg}}$ assigned to Ecobulk unit $j$:

$$\mathrm{Tf}_{i,n} - \mathrm{Ts}_{i,n} \le 720 + \max(0,\, D^{\max}_i - 720)\,(1 - y_{i,j,n})$$

#### Two-Path Intermediate Stg Storage

For lines $l \in L^{\mathrm{stg}}$, wine may optionally pass through an intermediate holding tank (Stg) before barrel or jar aging. This creates two parallel aging paths that both produce the same downstream state $\mathrm{va}_l$:

- **Direct path:** $\mathrm{Fl}_l \to \mathrm{vl}_l \to \mathrm{AgeDirect}_l \to \mathrm{va}_l \to \mathrm{Cs}_l$
- **Stg path:** $\mathrm{Fl}_l \to \mathrm{vl}_l \to \mathrm{Stg}_l \to \mathrm{vbuf}_l \to \mathrm{AgeViaStg}_l \to \mathrm{va}_l \to \mathrm{Cs}_l$

Stg is optional: the solver activates it only when it is economically beneficial.

**Buffer state.** $\mathrm{vbuf}_l \in S^{\mathrm{vbuf}} \subset S^{\mathrm{ZW}}$. Zero-wait classification forces $\mathrm{ST}_{\mathrm{vbuf}_l,n}=0$ at every event and the tight timing constraint:

$$W_{\mathrm{Stg}_l,n} \land W_{\mathrm{AgeViaStg}_l,n+1} \implies \mathrm{Ts}_{\mathrm{AgeViaStg}_l,n+1} = \mathrm{Tf}_{\mathrm{Stg}_l,n}$$

AgeViaStg must begin exactly when Stg ends, no dwell in an untracked intermediate vessel.

**Mandatory coupling (tc1).** If Stg runs, the Stg-path aging variant must run at the next event:

$$W_{\mathrm{AgeViaStg}_l,n+1} = W_{\mathrm{Stg}_l,n} \quad \forall l \in L^{\mathrm{stg}},\ n < N$$

This prevents discarding an entire batch that was already held in Stg (which would make the holding step pointless). Small capacity-granularity discards (e.g. 750 L not fitting into a whole number of 250 L barriques) are still absorbed by $\mathrm{Discard}_{\mathrm{vbuf}_l,n}$ via the zero-wait material balance:

$$b_{\mathrm{AgeViaStg}_l,n+1} + \mathrm{Discard}_{\mathrm{vbuf}_l,n+1} = b_{\mathrm{Stg}_l,n} \cdot \rho^{\mathrm{prod}}_{\mathrm{Stg}_l,\mathrm{vbuf}_l}$$

**Variable Stg duration.** Stg holding tasks ($i \in I^{\mathrm{stg}}$) have a variable duration: a lower bound rather than the fixed-duration equality of processing tasks. This is expressed by the global `duration_stor_lb/ub` constraints (see [Batch and Duration Linking](#batch-and-duration-linking)), not inside the disjunct:

$$\mathrm{Ts}_{\mathrm{Stg}_l,n} + \alpha_{\mathrm{Stg}} W_{\mathrm{Stg}_l,n} \;\le\; \mathrm{Tf}_{\mathrm{Stg}_l,n} \;\le\; \mathrm{Ts}_{\mathrm{Stg}_l,n} + D^{\max}_{\mathrm{Stg}_l} W_{\mathrm{Stg}_l,n} \quad (\alpha_{\mathrm{Stg}} = 48\text{ h})$$

The solver is free to extend the hold between the minimum $\alpha_{\mathrm{Stg}}$ and the time-window width $D^{\max}_{\mathrm{Stg}_l} = \mathrm{LF}_{\mathrm{Stg}_l} - \mathrm{ES}_{\mathrm{Stg}_l}$. The ecobulk cap (see above) provides a tighter upper bound when the task runs in an ecobulk unit.

**Rho overrides for AgeViaStg.** The Stg-variant aging task consumes the buffer state rather than the pre-aging filtration state:

$$\rho^{\mathrm{cons}}_{\mathrm{AgeViaStg}_l,\mathrm{vl}_l} = 0, \quad \rho^{\mathrm{cons}}_{\mathrm{AgeViaStg}_l,\mathrm{vbuf}_l} = -1, \quad \rho^{\mathrm{prod}}_{\mathrm{AgeViaStg}_l,\mathrm{va}_l} = \rho^{\mathrm{prod}}_{\mathrm{AgeDirect}_l,\mathrm{va}_l}$$

**tc1 skips for stg lines.** The one-to-one coupling (A13a) is intentionally omitted for the following step pairs on stg lines, as the state types and the Stg-specific coupling above handle ordering:

| Skipped pair                              | Reason                                                  |
|-------------------------------------------|---------------------------------------------------------|
| $\mathrm{Fl}_l \to \mathrm{Stg}_l$        | Stg is optional                                         |
| $\mathrm{Stg}_l \to \mathrm{AgeDirect}_l$ | Direct path is independent of Stg                       |
| $\mathrm{AgeDirect}_l \to \mathrm{Cs}_l$  | NIS on $\mathrm{va}_l$ enforces ordering for both paths |
| $\mathrm{AgeViaStg}_l \to \mathrm{Cs}_l$  | Same: NIS on $\mathrm{va}_l$                            |

**Updated aging throughput link (`A18_aging_link`).** For lines with Stg, both aging tasks contribute to final production:

$$\mathrm{FinalProd}_s \le \sum_{t \in \{\mathrm{AgeDirect}_s,\mathrm{AgeViaStg}_s\}} \sum_{n \in N} b_{t,n} \quad \forall s \in S^{\mathrm{Market}} : l_s \in L^{\mathrm{stg}}$$

For lines without Stg the original single-task bound applies unchanged.

#### Static Preprocessing (Event-Range Fixing)

In an event-based formulation with $N$ events and a production line of $K$ ordered steps, the $k$-th task can only be feasibly scheduled at events $k$ through $N-(K-k)$. Tasks outside this window are provably inactive and their disjunct indicators are fixed to 0 before the solve:

$$ W_{i_k,n} = 0 \quad \forall\, n < k \;\text{ or }\; n > N-(K-k) $$

#### Batch and Duration Linking

Three constraint groups that the makespan GDP model leaves inside the task disjunction are lifted out into global linear constraints keyed on the activation binary $W_{i,n}$. The motivation is that `gdp.hull` blows up on this model (the disjuncts share many continuous variables,  $\mathrm{Ts}, \mathrm{Tf}, b, b_{i,j,n}, \mathrm{Tsj}, \mathrm{Tfj}$, over wide ranges, and the active disjunct nests a second unit-selection layer), so the model is solved with `gdp.bigm` and the relaxation is strengthened manually instead.

**Batch on/off link (`b_on_off`).** A direct link tying the batch to activation:

$$ b_{i,n} \le \bar{B}_i \, W_{i,n} \quad \forall i \in I,\ n \in N $$

where $\bar{B}_i$ is the per-task batch upper bound (the sum of the $j^i_{\mathrm{max}}$ largest compatible-unit capacities, capped by any product upper bound; for pool tasks $\bar{B}_i = \bar{n}_{\mathrm{pool}} \cdot B_{\mathrm{pool}}$). This also ties the pool tasks (barrique/jar/ecobulk) directly to $W$, so fractional $W$ can no longer carry batch for free.

**Global duration link (`duration_proc`, `duration_stor_lb/ub`).**

Processing tasks ($i \notin I^{\mathrm{stg}}$):

$$ \mathrm{Tf}_{i,n} = \mathrm{Ts}_{i,n} + \alpha_i W_{i,n} + \beta_i b_{i,n} $$

Storage / holding tasks ($i \in I^{\mathrm{stg}}$, variable duration):

$$ \mathrm{Ts}_{i,n} + \alpha_i W_{i,n} \;\le\; \mathrm{Tf}_{i,n} \;\le\; \mathrm{Ts}_{i,n} + D^{\max}_i W_{i,n}, \qquad D^{\max}_i = \mathrm{LF}_i - \mathrm{ES}_i $$

**Global idle-collapse (`idle_collapse`).** The symmetry-breaking collapse of an idle event's start time onto the previous finish (kept inside the inactive disjunct in the makespan model) is likewise removed from the GDP block, with a tight $M = D^{\max}_i$ instead of $H$. Together with `g06` ($\mathrm{Ts}_{i,n} \ge \mathrm{Tf}_{i,n-1}$):

$$ \mathrm{Ts}_{i,n} \le \mathrm{Tf}_{i,n-1} + D^{\max}_i\, W_{i,n} \qquad \forall i \in I,\ n > 1 $$

$W_{i,n}=0$ gives $\mathrm{Ts}_{i,n} = \mathrm{Tf}_{i,n-1}$ (the same collapse). $W_{i,n}=1$ relaxes by the window width, not $H$. $M = D^{\max}_i$ is valid since $\mathrm{Tf}_{i,n-1} \ge \mathrm{ES}_i$ and $\mathrm{Ts}_{i,n} \le \mathrm{LF}_i$.

#### Identical-Unit Symmetry Breaking

Several units have multiple interchangeable physical instances (`inox_ext_10000` x2, `inox_ext_5000v` x2, `ecobulk_1100` x10). Permuting the instance labels of one base type leaves every constraint unchanged (identical capacity, flags, and bounds), so the solver would otherwise explore equivalent assignments. The instances $u_1,\dots,u_k$ of each multi-instance base are ordered by total usage over the schedule:

$$\sum_{i,n} y_{i,u_{m+1},n} \;\le\; \sum_{i,n} y_{i,u_m,n} \qquad m = 1,\dots,k-1$$

#### Remaining Economic Constraints

Final produced quantity is linked to last-event inventory and production (A18):

$$ \mathrm{ST}_{s,N} + \sum_{i \in \mathrm{IPS}_s} \rho^{\mathrm{prod}}_{i,s} \, b_{i,N} = \mathrm{FinalProd}_s \quad \forall s \in S^{\mathrm{P}} $$

**Aging throughput link (`A18_aging_link`):**
For products that have an aging step, all in-house production must pass through it. The total aging batch across all events upper-bounds $\mathrm{FinalProd}_s$. For lines without Stg (single aging task):

$$ \mathrm{FinalProd}_s \le \sum_{n \in N} b_{\mathrm{AgeDirect}_s,n} \quad \forall s \in S^{\mathrm{Market}} : l_s \notin L^{\mathrm{stg}} $$

For lines with Stg, both the direct and Stg-path aging variants contribute:

$$ \mathrm{FinalProd}_s \le \sum_{t \in \{\mathrm{AgeDirect}_s,\mathrm{AgeViaStg}_s\}} \sum_{n \in N} b_{t,n} \quad \forall s \in S^{\mathrm{Market}} : l_s \in L^{\mathrm{stg}} $$

**Per-product lateness (`LateDefByProduct`):**
Products with an aging stage are exempt from the lateness penalty entirely. Their delivery date is considered flexible because the aging duration is inherent to the product and not under the scheduler's control.

For non-aging products, each batch of the last mandatory task $i^{\mathrm{last}}_l$ must be assigned to one of the $K$ campaigns, and lateness is measured against the corresponding campaign deadline $D_k$. Assignment is encoded via a nested GDP disjunction inside the active-task disjunct:

$$\forall l \notin I^{\mathrm{Age}},\ n \in N:\quad W_{i^{\mathrm{last}}_l,n} \implies \bigvee_{k=1}^{K_l} \bigl[\mathrm{Tf}_{i^{\mathrm{last}}_l,n} \le D_k + \mathrm{Late}_{s_l,k}\bigr]$$

where $K_l = i_{\mathrm{max,young}}$ for young wine lines and $K_l = 1$ for all other non-aging lines. For $K_l = 1$ the disjunction collapses to a single Big-M inequality:

$$\mathrm{Tf}_{i^{\mathrm{last}}_l,n} \le D_1 + \mathrm{Late}_{s_l,1} + M^{\mathrm{late}}(1 - W_{i^{\mathrm{last}}_l,n})$$

**Campaign ordering.** To prevent two batches of the same young wine line from being assigned the same campaign label, the campaign index of each active event point is constrained to equal one plus the number of earlier unfixed active event points. For event $n$ with $S$ unfixed active predecessors $n' < n$, the single equality

$$\sum_{k=2}^{K_l} (k-1)\,c_k = S$$

(where $c_k$ is the binary indicator of the $k$-th campaign disjunct at event $n$) encodes the XOR constraint: given $c_1+\ldots+c_{K_l}=1$, the unique feasible solution is $c_{S+1}=1$. Campaign disjuncts with index $k > S+1$ are fixed inactive before the solve.

This equality constraint is added as a component of the **active disjunct** $d_{\mathrm{active}}$ for event $n$, not as a global algebraic constraint. When added globally, the GDP transformation zeros all nested campaign indicator variables $c_k$ whenever $W_{i,n}=0$, making the left-hand side 0 while $S \ge 1$ (from prior active events), yielding $0 = S$ which is infeasible. Placing the constraint inside $d_{\mathrm{active}}$ ensures it is only enforced when the task is actually running at event $n$.

### Economic Objective

Revenue from committed sales:

$$ \mathrm{Revenue} = \sum_{s\in S^{\mathrm{Market}}} \mathrm{Price}_s \cdot D_s $$

The model treats demand as committed sales, so this revenue term is constant with respect to scheduling decisions.

Grape-skin by-product revenue collected during pressing tasks:

$$ \mathrm{GrapeSkinRevenue} = \sum_{i \in I^{\mathrm{press}}} \sum_{n \in N} c^{\mathrm{skin}}_i \cdot b_{i,n} $$

Outsourcing cost:

$$ \mathrm{OutsourcingCost} = \sum_{s\in S^{\mathrm{Market}}} C^{\mathrm{out}}_s \cdot \mathrm{Outsource}_s $$

Lateness cost (summed over all campaigns and non-aging products):

$$\mathrm{LatenessCost} = c^{\mathrm{late}} \cdot \sum_{k=1}^{K} \sum_{s \in S^{\mathrm{late}}_k} \mathrm{Late}_{s,k}$$

where $S^{\mathrm{late}}_1 = S^{\mathrm{Market}} \setminus S^{\mathrm{Age}}$ and $S^{\mathrm{late}}_k = S^{\mathrm{young}}$ for $k > 1$ (young wine products only, i.e. products whose line has no aging stage).

Raw material cost:

$$ \mathrm{RawMaterialCost} = \sum_{s\in S^{\mathrm{R}}}\sum_{i:(i,s)\in \mathrm{ICS}}\sum_{n\in N} C_s^{\mathrm{raw}}\left(-\rho^{\mathrm{cons}}_{i,s}\right) b_{i,n} $$

Discard cost:

$$ \mathrm{DiscardCost} = c^{\mathrm{discard}} \sum_{s \in S^{\mathrm{I}}} \sum_{n \in N} \mathrm{Discard}_{s,n} $$

The economic profit groups all market-facing terms:

$$ \mathrm{Profit} = \mathrm{Revenue} + \mathrm{GrapeSkinRevenue} - \mathrm{OutsourcingCost} - \mathrm{RawMaterialCost} - \mathrm{LatenessCost} - \mathrm{DiscardCost} $$

Cooling cost applies to non-storage tasks ($i \in I^{\mathrm{nst}}$) running in exterior units ($j \in J^{\mathrm{ext}}$). The model validates at build time that $\beta_i = 0$ for all exterior-unit tasks (a `ValueError` is raised if any such task has $\beta_i \ne 0$), so cooling time reduces to the fixed duration $\alpha_i$.

The disjunctive batch bounds force $b_{i,j,n} = 0$ whenever $y_{i,j,n} = 0$ (assign disjunct sets $b_{i,j,n} \in [B^{\mathrm{min}}_{i,j}, B^{\mathrm{max}}_{i,j}]$, skip disjunct sets $b_{i,j,n} = 0$). Therefore $b_{i,j,n}$ already carries the on/off state and the cooling cost is exactly linear in $b_{i,j,n}$:

$$ \mathrm{CoolingCost} = c^{\mathrm{cool}} \sum_{\substack{(i,j) \in \mathrm{IJ} \\ j \in J^{\mathrm{ext}} \\ i \in I^{\mathrm{nst}}}} \sum_{n \in N} \alpha_i \, b_{i,j,n} $$

This is equivalent to the McCormick envelope of $\alpha_i \, b_{i,j,n} \, W_{i,n}$ but adds zero auxiliary variables and zero structural constraints, tightening the LP relaxation.

Storage cooling associated with $\mathrm{Stg}$ tasks ($i \in I^{\mathrm{st}}$) is intentionally excluded from the objective in the current formulation.

**Degeneracy-breaking penalty.** A small $\varepsilon$ term pulls all task start times as early as possible, breaking the degeneracy of floating tasks (tasks that can shift freely within a feasible window without changing the economic objective):

$$ \mathrm{TimePullPenalty} = \varepsilon \sum_{i \in I} \sum_{n \in N} \mathrm{Ts}_{i,n}, \quad \varepsilon = 10^{-6} $$

The coefficient is small enough that it never alters the economic ranking of solutions.

Full profit-maximization objective:

$$ \max Z = \mathrm{GrapeSkinRevenue} - \mathrm{OutsourcingCost} - \mathrm{LatenessCost} - \mathrm{RawMaterialCost} - \mathrm{DiscardCost} - c^{\mathrm{MS}} \cdot \mathrm{MS} - \mathrm{CoolingCost} - \mathrm{TimePullPenalty} $$

### Barrique and Jar Pool Aggregation

Barrique and jar vessels are pooled rather than tracked individually. Instead of one binary $y_{i,j,n}$ per vessel, an integer count variable replaces the entire pool.

**Additional sets:**

- $I^{\mathrm{Bar}} \subseteq I$: Barrique aging tasks
- $I^{\mathrm{Jar}} \subseteq I$: Jar aging tasks

**Additional parameters:**

- $\bar{n}_{\mathrm{bar}}$: Total number of barrique vessels in the pool
- $\bar{n}_{\mathrm{jar}}$: Total number of jar vessels in the pool
- $B_{\mathrm{bar}}$: Fixed capacity of one barrique (L)
- $B_{\mathrm{jar}}$: Fixed capacity of one jar (L)

**Additional variables:**

- $\mathrm{NumBarr}_{i,n} \in \mathbb{Z}_{\ge 0}$: Number of barriques used by task $i$ at event $n$; $\mathrm{NumBarr}_{i,n} \le \bar{n}_{\mathrm{bar}}$
- $\mathrm{NumJar}_{i,n} \in \mathbb{Z}_{\ge 0}$: Number of jars used by task $i$ at event $n$; $\mathrm{NumJar}_{i,n} \le \bar{n}_{\mathrm{jar}}$

Batch size for pool tasks is determined by the vessel count and fixed capacity:

$$b_{i,n} = \mathrm{NumBarr}_{i,n} \cdot B_{\mathrm{bar}} \quad \forall i \in I^{\mathrm{Bar}}$$

$$b_{i,n} = \mathrm{NumJar}_{i,n} \cdot B_{\mathrm{jar}} \quad \forall i \in I^{\mathrm{Jar}}$$

**Activation-linked vessel-count bounds.** When a pool task is active, vessel counts are bounded by per-task minimum units and by pool size:

$$\mathrm{NumBarr}_{i,n} \ge j^{i}_{\mathrm{min}}\,W_{i,n} \quad \forall i \in I^{\mathrm{Bar}}$$

$$\mathrm{NumJar}_{i,n} \ge j^{i}_{\mathrm{min}}\,W_{i,n} \quad \forall i \in I^{\mathrm{Jar}}$$

$$\mathrm{NumBarr}_{i,n} \le \bar{n}_{\mathrm{bar}}\,W_{i,n} \quad \forall i \in I^{\mathrm{Bar}}$$

$$\mathrm{NumJar}_{i,n} \le \bar{n}_{\mathrm{jar}}\,W_{i,n} \quad \forall i \in I^{\mathrm{Jar}}$$

**Per-event-index pool capacity.** At each event point, the total vessels in use across all pool tasks cannot exceed the pool size:

$$\sum_{i \in I^{\mathrm{Bar}}} \mathrm{NumBarr}_{i,n} \le \bar{n}_{\mathrm{bar}} \quad \forall n \in N$$

$$\sum_{i \in I^{\mathrm{Jar}}} \mathrm{NumJar}_{i,n} \le \bar{n}_{\mathrm{jar}} \quad \forall n \in N$$

This handles within-event conflicts. Cross-event conflicts (tasks at different event indices that overlap in time) are handled by the cross-event sequencing formulation below.

**Cross-event sequencing.** The per-vessel timing constraint A09 ($\mathrm{Tsj}_{j,n+1} \ge \mathrm{Tfj}_{j,n}$) and the pool capacity constraint have no direct equivalent when vessels are not tracked individually. They are replaced by a cross-event sequencing formulation.

After static preprocessing (event-range fixing), let $\mathcal{P}^{\mathrm{Bar}}$ be the set of unordered pairs of barrique task-events that are not provably inactive **and** for which temporal overlap is not already ruled out by other constraints:

```math
\mathcal{P}^{\mathrm{Bar}} = \bigl\{\{(i_1,n_1),(i_2,n_2)\} : i_1,i_2 \in I^{\mathrm{Bar}},\ \text{neither fixed inactive},\ n_1 \ne n_2,\ i_1 \ne i_2,\ \mathrm{LF}_{i_1} > \mathrm{ES}_{i_2},\ \mathrm{LF}_{i_2} > \mathrm{ES}_{i_1}\bigr\}
```

Three pruning rules are applied to this index set before constructing the sequencing binaries:

| Pruned pair                                                                            | Reason                                                                                                                                |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| $n_1 = n_2$                                                                            | Within-event capacity bound $\sum_i \mathrm{NumBarr}_{i,n} \le \bar{n}_{\mathrm{bar}}$ already enforces non-overlap at this event     |
| $i_1 = i_2$                                                                            | Same-task event chain $g06$ ($\mathrm{Ts}_{i,n+1} \ge \mathrm{Tf}_{i,n}$) forces strict temporal ordering, so no concurrent occupancy |
| $\mathrm{LF}_{i_1} \le \mathrm{ES}_{i_2}$ or $\mathrm{LF}_{i_2} \le \mathrm{ES}_{i_1}$ | One direction of ordering is already forced by the time-window bounds, so the pair cannot overlap                                     |

For each remaining pair, two sequencing binaries encode the temporal ordering:

- $z^{\mathrm{fwd}}_{i_1,n_1,i_2,n_2} \in \{0,1\}$: 1 if task-event $(i_1,n_1)$ finishes before $(i_2,n_2)$ starts
- $z^{\mathrm{rev}}_{i_1,n_1,i_2,n_2} \in \{0,1\}$: 1 if task-event $(i_2,n_2)$ finishes before $(i_1,n_1)$ starts

When both are 0, the two task-events are concurrent.

**Forward sequencing:**

$$\mathrm{Tf}_{i_1,n_1} \le \mathrm{Ts}_{i_2,n_2} + M^{\ge}_{i_1,i_2}\bigl(1 - z^{\mathrm{fwd}}_{i_1,n_1,i_2,n_2}\bigr)$$

**Reverse sequencing:**

$$\mathrm{Tf}_{i_2,n_2} \le \mathrm{Ts}_{i_1,n_1} + M^{\ge}_{i_2,i_1}\bigl(1 - z^{\mathrm{rev}}_{i_1,n_1,i_2,n_2}\bigr)$$

**Pool capacity (cross-event overlap coupling; complements per-event bounds and replaces per-vessel A09 for pooled units):**

$$\mathrm{NumBarr}_{i_1,n_1} + \mathrm{NumBarr}_{i_2,n_2} \le \bar{n}_{\mathrm{bar}} + \bar{n}_{\mathrm{bar}}\bigl(z^{\mathrm{fwd}}_{i_1,n_1,i_2,n_2} + z^{\mathrm{rev}}_{i_1,n_1,i_2,n_2}\bigr)$$

When both sequencing binaries are 0 (concurrent), this reduces to $\mathrm{NumBarr}_{i_1,n_1} + \mathrm{NumBarr}_{i_2,n_2} \le \bar{n}_{\mathrm{bar}}$. When either is 1 (sequential), the right-hand side is relaxed to $2\bar{n}_{\mathrm{bar}}$, so no additional cross-event coupling is needed for that pair. The per-event bounds $\sum_i \mathrm{NumBarr}_{i,n} \le \bar{n}_{\mathrm{bar}}$ and $\sum_i \mathrm{NumJar}_{i,n} \le \bar{n}_{\mathrm{jar}}$ remain active.

The same formulation applies to $\mathcal{P}^{\mathrm{Jar}}$ with $\mathrm{NumJar}$, $\bar{n}_{\mathrm{jar}}$, $z^{\mathrm{fwd/rev}}$ replaced by their jar counterparts.

**Maximum idle time (BarriqueMaxIdle).** Barriques must be refilled within 24 hours of being emptied. When $(i_1,n_1)$ precedes $(i_2,n_2)$:

$$\mathrm{Ts}_{i_2,n_2} \le \mathrm{Tf}_{i_1,n_1} + 24 + M^{\mathrm{max}}_{i_1,i_2}\bigl(1 - z^{\mathrm{fwd}}_{i_1,n_1,i_2,n_2}\bigr)$$

$$\mathrm{Ts}_{i_1,n_1} \le \mathrm{Tf}_{i_2,n_2} + 24 + M^{\mathrm{max}}_{i_2,i_1}\bigl(1 - z^{\mathrm{rev}}_{i_1,n_1,i_2,n_2}\bigr)$$

Since all vessels in a task-event share a single start time $\mathrm{Ts}_{i,n}$, this constraint on the task-event level is equivalent to the per-vessel constraint: any vessel reused between two sequential task-events must be refilled within 24 hours, and because all vessels in the second task start at $\mathrm{Ts}_{i_2,n_2}$ simultaneously, the constraint cannot be relaxed for a subset of vessels.
