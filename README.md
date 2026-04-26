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
      - [Ecobulk Alm Maximum Duration (TODO)](#ecobulk-alm-maximum-duration-todo)
      - [Static Preprocessing (Event-Range Fixing)](#static-preprocessing-event-range-fixing)
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

- $I^{st} \subset I$: Storage tasks
- $I^{nst} \subset I$: Non-storage tasks
- $I^{pst} \subset I$: Pre-storage tasks
- $J^{st} \subset J$: Storage units
- $S^{p} \subset S$: Final products
- $S^{zw} \subset S$: Zero-wait states (includes shared must pool $m$)
- $S^{nis} \subset S$: No-intermediate-storage states

This lets the same formulation handle very different behaviors just by set membership.

## Parameters

- $\alpha_i$: Fixed processing time for task $i$
- $\beta_i$: Variable processing time coefficient for task $i$
- $\rho_{i,s}^{prod}$: Production coefficient of state $s$ by task $i$
- $\rho_{i,s}^{cons}$: Consumption coefficient of state $s$ by task $i$
- $B_{i,j}^{min}, B_{i,j}^{max}$: Min/Max batch size for task $i$ in unit $j$
- $H$: Time horizon
- $D_s$: Demand for state $s$

Activation and unit-assignment limits:

- $i_{max}$: Default maximum number of activations for non-storage tasks (typically 1)
- $task_{i_{max,i}}$: Per-task activation limit. Defaults to $i_{max}$, overridden for the shared pressing task to equal the number of pressing lines
- $j_{max}$: Maximum number of units assigned to an active non-storage task at an event (overridden per task via `max_units` in the template)
- $j_{min}$: Minimum number of units assigned to an active non-storage task at an event
- $i_{max}^{st}$: Maximum activation indicator for storage tasks (applied at final event)
- $j_{max}^{st}$: Maximum number of units assigned to an active storage task at an event
- $j_{min}^{st}$: Minimum number of units assigned to an active storage task at an event

## Decision Variables

- $W_{i,n} \in \{0,1\}$: 1 if task $i$ is active in event $n$ (does task i occur at event n?)
- $y_{i,j,n} \in \{0,1\}$: 1 if task $i$ uses unit $j$ in event $n$ (if it occurs, which unit does it use?)
- $b_{i,n} \ge 0$: Batch size of task $i$ in event $n$ (total amount of wine processed in that event?)
- $b_{i,j,n} \ge 0$: Batch size of task $i$ in unit $j$ in event $n$ (how much wine is processed in that unit?)
- $ST_{s,n} \ge 0$: Inventory of state $s$ at event $n$ (how much material exists after each event)
- $Ts_{i,n}, Tf_{i,n} \ge 0$: Start and finish times of task $i$ in event $n$
- $Tsj_{j,n}, Tfj_{j,n} \ge 0$: Start and finish times of unit $j$ in event $n$
- $MS$: Makespan (total schedule length)

## Equations

### 1. Material Balances

**Initial Event ($n=1$) (h04):**

$$ ST_{s,1} = ST0_s + \sum_{i \in ICS_s} \rho_{i,s}^{cons} b_{i,1} $$

Inventory starts from an initial stock and subtracts what is consumed at event 1.

**Subsequent Events ($n > 1$) (h03):**

$$ ST_{s,n} = ST_{s,n-1} + \sum_{i \in IPS_s} \rho_{i,s}^{prod} b_{i,n-1} + \sum_{i \in ICS_s} \rho_{i,s}^{cons} b_{i,n} $$

Inventory evolves as:

- previous inventory
- plus production from tasks at the previous event
- minus consumption from tasks at the current event

### 2. Task Timing and Sequencing

**Processing Time (h05):**
For non-storage tasks ($i \in I^{nst}$):

$$ Tf_{i,n} = Ts_{i,n} + \alpha_i W_{i,n} + \beta_i b_{i,n} $$

- `duration = fixed time + batch-dependent time`
- multiplied by activation `W`
- If the task is inactive, duration is equal to zero.

**Task Sequence (g06):**

$$ Ts_{i,n+1} \ge Tf_{i,n} $$

A task at event `n+1` cannot start before it finishes at `n`.
This creates a natural temporal chain across events.

**Task Precedence (g11):**
If task $i$ consumes what task $i'$ produces:

$$ Ts_{i,n+1} \ge Tf_{i',n} - H(1 - W_{i',n}) $$

- `i` at `n+1` must start after `i′` at `n` finishes
- Big-M deactivates the constraint when tasks are inactive

**Zero Wait (ZW) Constraint (g12):**
For states $s \in S^{zw}$, if $i$ consumes $s$ from $i'$:

$$ Ts_{i,n+1} \le Tf_{i',n} + H(2 - W_{i',n} - W_{i,n+1}) $$

Zero-wait: consumption must start exactly when production ends

**No Intermediate Storage (NIS) Constraint (g13):**
For states $s \in S^{nis}$, if $i$ consumes $s$ from $i'$:

$$ Ts_{i,n+1} \le Tf_{i',n} + H(2 - W_{i',n} - W_{i,n+1}) $$

No-intermediate-storage: material cannot sit in inventory

### 3. Unit Assignment and Capacity

Total time spent by all tasks on a unit cannot exceed the horizon H. This prevents hidden overlaps.

Each task:

- needs at least $j_{min}$ units
- can use at most $j_{max}$ units
- only if the task is active

This supports parallel units (e.g. multiple tanks for one blend).

**Unit Capacity Horizon (h09):**

$$ \sum_{i \in I^{nst}} \sum_{n} (\alpha_i y_{i,j,n} + \beta_i b_{i,j,n}) \le H $$

**Max Units per Task (A01, A01st):**

$$ \sum_{j} y_{i,j,n} \le j_{max} \cdot W_{i,n} $$

**Min Units per Task (A02, A02st):**

$$ \sum_{j} y_{i,j,n} \ge j_{min} \cdot W_{i,n} $$

**Single Task per Unit (A03):**

$$ \sum_{i} y_{i,j,n} \le 1 $$

Each unit can handle at most one task per event.

**Shared pressing task.** All pressing lines (white and rosé) share a single `Pr` task and a single zero-wait must pool state $m$. The `Pr` task consumes grapes from the shared raw-material pool and produces into $m$; each fermentation task for a pressed wine consumes from $m$. This eliminates artificial per-line symmetry: the press can be scheduled once and serve multiple fermentation tasks at the next event. The per-task activation limit is tightened to

$$task_{i_{max,Pr}} = \min\!\bigl(n_{press},\ n_{max} - (K_{min} - 1)\bigr)$$

where $n_{press}$ is the number of pressing lines and $K_{min}$ is the step count of the shortest pressing line. The second term caps `Pr` by the number of events that can precede the final downstream step: a press run at event $n$ requires at least $K_{min}-1$ more events after it, so `Pr` can feasibly start at most $n_{max}-(K_{min}-1)$ times. In the makespan models (where outsourcing is unavailable) a complementary lower bound is also imposed.

### 4. Batch Sizing

**Batch Aggregation (A04):**

$$ b_{i,n} = \sum_{j} b_{i,j,n} $$

Total batch size equals the sum over assigned units.

**Batch Limits (A05a, A05b):**

$$ B_{i,j}^{min} y_{i,j,n} \le b_{i,j,n} \le B_{i,j}^{max} y_{i,j,n} $$

If a unit is used, batch size must lie between min and max.
If it is not used, batch size is forced to zero.

**Fixed-capacity tightening.** When $B_{i,j}^{min} = B_{i,j}^{max} = B$ (most subterranean tanks, barriques, jars, ecobulks), the two inequalities collapse to one equality inside the assign disjunct:

$$ b_{i,j,n} = B \cdot y_{i,j,n} $$

### 5. Operational Constraints

**Max Activations - Non-storage tasks (A06):**

$$ \sum_{n} W_{i,n} \le task_{i_{max,i}} \quad \forall i \in I^{nst} $$

Limits how many times a non-storage task can occur. For most tasks $task_{i_{max,i}} = i_{max} = 1$ (one fermentation per batch). The shared pressing task uses the tighter bound $task_{i_{max,Pr}} = \min(n_{press},\ n_{max}-(K_{min}-1))$ as described above.

**Min Activations of Pr - Makespan models (A06_min_Pr):**

In the makespan models all demand must be met in-house (no outsourcing), so the total must produced by the press must equal at least the sum of each pressing line's demand divided by its chain yield from must to final product:

$$ \sum_{n} W_{Pr,n} \ge \left\lceil \frac{\displaystyle\sum_{l \in L_{press}} D_{s_l}\,/\,\eta_l}{\max_j B^{max}_{Pr,j} \cdot \rho^{prod}_{Pr,m}} \right\rceil $$

where $\eta_l = \prod_k \rho^{prod}_k$ is the chain yield from must to final product along line $l$ and $\rho^{prod}_{Pr,m}$ is the must yield of the press.

**Max Activations - Storage tasks (A06st):**

$$ W_{i,N} \le i_{max}^{st} \quad \forall i \in I^{st} $$

With the persistence chain (A14, see below), $W_{i,n}$ is non-decreasing, so the value at the final event $N$ counts whether the task ever started.

**Storage Task Duration (A07, A08):**

$$ Tf_{i,n} \ge Ts_{i,n} $$

$$ Tf_{i,n} \ge H \cdot W_{i,n} $$

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

$$ Tsj_{j,n+1} \ge Tfj_{j,n} $$

**Unit Duration (A10):**

$$ Tfj_{j,n} \ge Tsj_{j,n} $$

**Sync Start Times (A11a, A11b):**

$$ Tsj_{j,n} \le Ts_{i,n} + H(1 - y_{i,j,n}) $$

$$ Tsj_{j,n} \ge Ts_{i,n} - H(1 - y_{i,j,n}) $$

**Sync End Times (A12a, A12b):**

$$ Tfj_{j,n} \le Tf_{i,n} + H(1 - y_{i,j,n}) $$

$$ Tfj_{j,n} \ge Tf_{i,n} - H(1 - y_{i,j,n}) $$

### 7. Special Constraints

**One-to-One Pairs (A13a):**

$$ W_{i',n+1} = W_{i,n} $$

For tightly coupled tasks (e.g. transfer immediately after fermentation), activation is forced to match across events.

**Storage Persistence (A14, A15):**
If storage task $i$ is active in $n$, it must remain active in $n+1$ (and by transitivity, all future events):

$$ W_{i,n+1} \ge W_{i,n} \quad \forall n \in \{1 \dots N-1\}, i \in I^{st} $$

$$ y_{i,j,n+1} \ge y_{i,j,n} \quad \forall n \in \{1 \dots N-1\}, (i,j) \in IJ, i \in I^{st} $$

Once a storage task starts:

- it must remain active in all future events
- same for unit assignment

This prevents emptying and reusing storage implicitly. The recursive chain is equivalent to the all-pairs formulation but uses $O(N)$ constraints instead of $O(N^2)$.

**Ecobulk Timing (A16):**
For states $s \in S^{FISEco}$ (Ecobulk compatible), if task $i$ consumes $s$ produced by task $i'$:

$$ Ts_{i,n+1} \le Tf_{i',n} + M(2 - W_{i',n} - W_{i,n+1}) $$

This enforces a "no intermediate storage" condition for Ecobulk products. The consuming task must start immediately after the producing task finishes (or before, effectively zero-wait when combined with precedence).

### 8. Final State and Objective

**Demand Satisfaction (g17):**

$$ FinalProd_s \ge D_s $$

Final production must meet or exceed demand.

**Final Production Calculation (A18):**

$$ ST_{s,N} + \sum_{i \in IPS_s} \rho_{i,s}^{prod} b_{i,N} = FinalProd_s $$

Final inventory plus last-event production equals total produced.

**Makespan (A21):**

$$ Tf_{i,N} \le MS \quad \forall i \in I^{pst} $$

The finish time of all pre-storage tasks must be ≤ MS.

**Objective Function:**

$$ \min Z = MS + Penalty_{unused} + Penalty_{space} $$

Unused-storage term:

$$ JST_{unused} = n_{JST} - \sum_{j \in JST}\sum_{\substack{i \in I^{st}:\\(i,j)\in IJ}} y_{i,j,N} $$

With storage persistence, this counts tanks that are inactive at the final event and is equivalent to tanks never used.

$$ Penalty_{unused} = c^{empty} \cdot \frac{1}{n_{JST}} \cdot JST_{unused} $$

$$ Penalty_{space} = c^{air} \cdot \frac{1}{\sum_{j\in JST}(\Delta B)/n_{JST}} \cdot \sum_{j\in JST} Freespace_j $$

Minimize makespan while trying to:

- penalize unused capacity
- reward efficient storage usage

## GDP Reformulation

The reformulation in this model (`WineSchedulingMS_GDP.py`) uses Disjunctive Programming to capture logical `OR` decisions (Run vs. Don't Run, Assign vs. Idle) directly, avoiding manual Big-M constraints for core operational logic.

### 1. Combined Task & Unit Disjunction (Eq. 2.1)

Task activation and unit selection are combined into a single nested disjunction. If a task is active it must pick its units; if inactive, everything is zeroed out.

$$\forall i \in I, n \in N:
\begin{bmatrix}
W_{i,n} \\
Tf_{i,n} = Ts_{i,n} + \alpha_i + \beta_i b_{i,n} \quad (i \in I^{nst}) \\
\bigvee_{j \in J_i} \begin{bmatrix}
y_{i,j,n} = 1 \\
B^{min}_{i,j} \le b_{i,j,n} \le B^{max}_{i,j} \\
Tsj_{j,n} = Ts_{i,n} \\
Tfj_{j,n} = Tf_{i,n}
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
Tf_{i,n} = Ts_{i,n} \\
\forall j \in J_i:
\begin{bmatrix}
y_{i,j,n} = 0 \\
b_{i,j,n} = 0
\end{bmatrix}
\end{bmatrix}$$

A global constraint enforces at most one task per unit per event:

$$ \sum_{i} y_{i,j,n} \le 1 \quad \forall j, n $$

**Symmetry breaking (inactive disjunct).** When a task is inactive at event $n > 1$, its start time is collapsed to the finish of the previous event:

$$\neg W_{i,n} \implies Ts_{i,n} = Tf_{i,n-1} \quad \forall i \in I,\ n > 1$$

This prevents $Ts_{i,n}$ from floating freely in $[0, H]$ when the task is idle, which otherwise creates flat LP plateaus and symmetric integer solutions.

**Variable time-window bounds.** In addition to using $ES_i$ and $LF_i$ in Big-M computation, they are applied as direct variable bounds before solving:

$$Ts_{i,n} \ge ES_i, \quad Tf_{i,n} \le LF_i, \quad Ts_{i,n} \le LF_i$$

### 2. Global Precedence Logic (Eq. 2.2)

Instead of 3-way disjunctions, precedence is expressed as conditional constraints using task-pair-specific Big-M implications on $W$.

For each producer-consumer pair $(i', i)$ linked through a shared state, define earliest start and latest finish times from minimum cumulative step durations:

$$ES_i = \sum_{k < \text{pos}(i)} \alpha_k, \qquad LF_i = H - \sum_{k > \text{pos}(i)} \alpha_k$$

These yield tight, pair-specific Big-M values:

$$M^\ge_{i',i} = \max(0,\ LF_{i'} - ES_i), \qquad M^\le_{i',i} = \max(0,\ LF_i - ES_{i'})$$

**General Precedence (`prec_ge`):**

$$ W_{i',n} \land W_{i,n+1} \implies Ts_{i,n+1} \ge Tf_{i',n} $$

Linearized as:

$$ Ts_{i,n+1} \ge Tf_{i',n} - M^\ge_{i',i}(2 - W_{i',n} - W_{i,n+1}) $$

**Zero-Wait / No-Intermediate-Storage / Ecobulk (equality when both active):**

$$ W_{i',n} \land W_{i,n+1} \implies Ts_{i,n+1} = Tf_{i',n} $$

Linearized as the $\ge$ above plus:

$$ Ts_{i,n+1} \le Tf_{i',n} + M^\le_{i',i}(2 - W_{i',n} - W_{i,n+1}) $$

Using $M_{i',i} \ll H$ for long-aging lines (up to 85% tighter than the global horizon) significantly strengthens the LP relaxation at the root node.

### 3. Storage Persistence via Logical Implications (Eq. 2.3)

Persistence rules for storage tasks are expressed as pure logical implications on disjunct indicator variables, rather than algebraic inequality chains:

$$ W_{i,n} \implies W_{i,n+1} \quad \forall i \in I^{st}, n < N $$

This means once a storage task activates at event $n$, it must remain active for all subsequent events. The unit-level persistence ($y_{i,j,n} \implies y_{i,j,n+1}$) is retained as an algebraic constraint (A15).

### 4. Algebraic Constraints (Global Coupling)

These constraints remain algebraic as they involve aggregation across the entire horizon or multiple units.

**Material Balances:**

$$ ST_{s,n} = ST_{s,n-1} + \sum \rho^{prod} b_{n-1} + \sum \rho^{cons} b_{n} $$

**Batch Aggregation (A04):**

$$ b_{i,n} = \sum_{j \in J_i} b_{i,j,n} $$

**Global Capacity (A06 / A06_min_Pr):**

$$ \sum_{n} W_{i,n} \le task\_i_{max,i} $$

For the shared pressing task, $task_{i_{max,Pr}}$ is bounded above by $\min(n_{press}, n_{max}-(K_{min}-1))$ and, in the makespan models, below by $\lceil TotalMust / MaxMustPerRun \rceil$.

**Unit Constraints:**

$$ \sum_{j} y_{i,j,n} \le j_{max} \cdot W_{i,n} $$

**Cover cuts (LP tightening).** After the disjunction structure is built, the following valid inequality is added for every non-pool task $(i, n)$:

$$b_{i,n} \le \sum_{j \in J_i} B^{max}_{i,j} \cdot y_{i,j,n}$$

This directly links the continuous batch variable to the binary unit-assignment variables without relying on the solver to derive it from the Big-M expansion of the disjunctive bounds.

### 5. Variable Definitions (GDP Adaptations)

- **$W_{i,n}$:** The GDP disjunct's `binary_indicator_var`, no separate binary variable is declared. This single binary serves both the disjunction structure and algebraic constraints (A01, A06, precedence, persistence, etc.), and activates terminal storage horizon sync inside the final storage disjunct.
- **$y_{i,j,n}$:** Unit assignment binary; retained for global constraints (A03, batch limits).

## Economic Optimization

This section documents the economic variant implemented in `WineSchedulingEconomic_GDP.py` using `parameters.toml`. The scheduling structure is the same GDP formulation described above, with added market variables, raw-material costs, aging logic, and a profit-maximizing objective.

### Additional Sets

- $S^{Market} \subset S^P$: Final product states sold on the market
- $S^R \subset S$: Shared raw-material pool states (e.g. $s_{red}$, $s_{white\_rose}$)
- $S^I \subset S$: Intermediate states (all states that are neither final products nor raw pools). The `Discard` variable is defined on this set.
- $S^{Age} \subset S$: Post-aging, pre-cold-stabilization intermediate states ($va_l$ for lines where aging precedes cold stabilization). Treated as NIS - cold stabilization must begin immediately after aging finishes.
- $I^{Age} \subseteq I$: Aging tasks ($AgeBar^*$, $AgeJar^*$). Each product $s$ has at most one aging task $i^{Age}_s$.
- $J^{ext} \subseteq J$: Exterior units that require active cooling (flagged via `exterior = true` in the TOML).

### Additional Parameters

- $Price_s$: selling price of product state $s$
- $C^{out}_s$: outsourcing cost of product state $s$
- $Deadline$: target completion time for all production lines
- $AgingHours_s$: fixed processing duration of the aging step for product $s$ (0 if the product has no aging step)
- $c^{late}$: lateness penalty coefficient (cost per hour per product)
- $c^{MS}$: makespan penalty coefficient (cost per hour of schedule length)
- $c^{cool}$: cooling cost per liter-hour for exterior tanks ($/(L \cdot h)$)
- $C_s^{raw}$: raw-material cost per liter for shared raw pools $s \in S^R$
- $c^{discard}$: cost per liter discarded from intermediate states
- $i_{max}^{young}$ (`iMaxYoungWine`): maximum activations per young wine line. This allows more than one campaing of young wines per campaing of aged wines. When $> 1$, allows multiple separate batches of young wines from the same line within a single schedule. Affects the `Pr` activation ceiling and all tasks in the pressing chain:

$$task_{i_{max,Pr}} = \min\bigl(n_{press} \cdot i_{max}^{young},\ n_{max} - (K_{min} - 1)\bigr)$$

$$task_{i_{max,i}} = i_{max}^{young} \quad \forall i \in \text{steps of pressing lines} \setminus \{Pr\}$$

### Additional Variables

- $Outsource_s \ge 0$: outsourced quantity of product $s$
- $LatenessProd_s \ge 0$: delay of product $s$ beyond its effective deadline
- $MS \ge 0$: makespan (retained from scheduling model)
- $FinalProd_s \ge 0$: final internally produced quantity of product state $s$
- $Discard_{s,n} \ge 0$: volume (L) discarded from intermediate state $s \in S^I$ at event $n$. Allows the consuming task's batch to be smaller than the upstream produced quantity when a fixed-capacity downstream vessel cannot accept the full volume.

### Economic Constraints

#### Material Balances (Economic Model)

The standard material balance (h03/h04) is extended with a `Discard` term on all intermediate states ($s \in S^I$):

**Initial event ($n=1$):**

$$ ST_{s,1} = ST0_s + \sum_{i \in ICS_s} \rho_{i,s}^{cons} b_{i,1} - Discard_{s,1} \quad \forall s \in S^I $$

**Subsequent events ($n > 1$):**

$$ ST_{s,n} = ST_{s,n-1} + \sum_{i \in IPS_s} \rho_{i,s}^{prod} b_{i,n-1} + \sum_{i \in ICS_s} \rho_{i,s}^{cons} b_{i,n} - Discard_{s,n} \quad \forall s \in S^I $$

For raw-material pools ($s \in S^R$) and final product states ($s \in S^P$), the original balances without `Discard` are retained.

**Effect on zero-wait / NIS states.** For states that enforce zero inventory ($ST_{s,n}=0$ always), the balance collapses to:

$$ \sum_{i \in ICS_s} \rho_{i,s}^{cons} b_{i,n} = \sum_{i \in IPS_s} \rho_{i,s}^{prod} b_{i,n-1} - Discard_{s,n} $$

Without `Discard`, the consuming batch is fully determined by the producing batch and the yield coefficients. With `Discard`, the constraint becomes $\le$. The consuming batch may be any fraction of the upstream volume and the excess is discarded. This allows the model to match fixed-capacity downstream vessels to variable upstream batches without infeasibility.

**Terminal balance (h16).** Zero-wait states require that any production at the final event $N$ (which has no successor event for a consuming task) is also explicitly accounted for. The constraint is extended to allow discarding this stranded volume:

$$ ST_{s,N} + \sum_{i \in IPS_s} \rho_{i,s}^{prod} b_{i,N} - Discard_{s,N} = 0 \quad \forall s \in S^{zw} $$

NIS states do not need an analogous terminal constraint: the NIS condition (`g13`) prevents any task producing into an NIS state from running at $N$ (it would require a consuming task at $N+1$ which does not exist), so production at the final event is always zero for these states.

#### Demand Constraint

Demand can be met by internal production plus outsourcing. For products with `OutsourceAllowed = 0`, the outsourcing term is blocked, so demand must be met entirely in-house:

$$ FinalProd_s + Outsource_s \ge D_s \quad \forall s \in S^{Market} $$

**Minimum-activation valid cuts (A06_min).** For each non-outsourceable line, the cut is applied to all non-press tasks in the line (not only the last task). For each such task $i$, at least
$\left\lceil D_{s_l}/\max_j B^{max}_{i,j}\right\rceil$ activations are required:

$$ \sum_{n \in N} W_{i,\, n} \ge \left\lceil \frac{D_{s_l}}{\displaystyle\max_{j:\,(i,j)\in IJ} B^{max}_{i,j}} \right\rceil \quad \forall l : s_l \notin Outsource,\ \forall i \in chain(l)\setminus\{Pr\} $$

**Shared pressing minimum-activation cut (A06_min_Pr).** In the economic model this cut is now added conditionally for non-outsource pressed lines, using the required total must volume and press yield (same structure as the makespan version, with outsourcing-aware filtering).

#### Zero-Inventory Constraints

Zero-wait and NIS states have $ST_{s,n} = 0$ enforced explicitly at every event as an algebraic constraint, not only through the material balance structure:

$$ST_{s,n} = 0 \quad \forall s \in S^{zw} \cup S^{nis},\ n \in N$$

These are added directly so the solver sees the equality without having to infer it from the interaction of h03/h04, the ZW/NIS precedence constraints, and the demand constraint.

#### Makespan Lower Bound

A valid lower bound on the makespan is derived from the minimum total processing time along the longest production line:

$$MS \ge \max_{l \in L} \sum_{k \in \text{steps}(l)} \alpha_k$$

This is the sum of fixed durations along each line (ignoring batch-dependent terms), taking the maximum over all lines.

#### Ecobulk Alm Maximum Duration (TODO)

Ecobulk tanks must not hold wine for more than 720 hours (1 month). For each Ecobulk-compatible Alm task $i$ assigned to Ecobulk unit $j$:

$$Tf_{i,n} - Ts_{i,n} \le 720 + H(1 - y_{i,j,n})$$

#### Static Preprocessing (Event-Range Fixing)

In an event-based formulation with $N$ events and a production line of $K$ ordered steps, the $k$-th task can only be feasibly scheduled at events $k$ through $N-(K-k)$. Tasks outside this window are provably inactive and their disjunct indicators are fixed to 0 before the solve:

$$ W_{i_k,\, n} = 0 \quad \forall\, n < k \;\text{ or }\; n > N-(K-k) $$

#### Remaining Economic Constraints

Final produced quantity is linked to last-event inventory and production (A18):

$$ ST_{s,N} + \sum_{i:(i,s)\in IPS} \rho^{prod}_{i,s} \, b_{i,N} = FinalProd_s \quad \forall s \in S^P $$

**Aging throughput link (`A18_aging_link`):**
For products that have an aging step, all in-house production must pass through it. Because the aging task is the sole producer of $va_s$ (for aging-first lines) or of the final product (for aging-last lines), the total aging batch across all events upper-bounds $FinalProd_s$:

$$ FinalProd_s \le \sum_{n \in N} b_{i^{Age}_s,\, n} \quad \forall s \in S^{Market} : i^{Age}_s \text{ exists} $$

**Per-product lateness (`LateDefByProduct`):**
Products with an aging stage are exempt from the lateness penalty entirely. Their delivery date is considered flexible because the aging duration is inherent to the product and not under the scheduler's control.

For non-aging products, lateness is the amount by which the finish time of the last mandatory task $i^{last}_l$ exceeds the deadline:

$$LatenessProd_{s_l} = \max(0,\ Tf_{i^{last}_l} - Deadline)$$

Implemented as:

$$ Tf_{i^{last}_l,\, n} \le Deadline + LatenessProd_{s_l} + M^{late}\,(1 - W_{i^{last}_l,\, n}) \quad \forall l \notin I^{Age},\, n \in N $$

### Economic Objective

Revenue from committed sales:

$$ Revenue = \sum_{s\in S^{Market}} Price_s \cdot D_s $$

The model treats demand as committed sales, so this revenue term is constant with respect to scheduling decisions.

Grape-skin by-product revenue collected during pressing tasks:

$$ GrapeSkinRevenue = \sum_{i \in I^{press}} \sum_{n \in N} c^{skin}_i \cdot b_{i,n} $$

Outsourcing cost:

$$ OutsourcingCost = \sum_{s\in S^{Market}} C^{out}_s \cdot Outsource_s $$

Lateness cost (summed over non-aging products only):

$$ LatenessCost = c^{late} \cdot \sum_{s \in S^{Market} \setminus S^{Age}} LatenessProd_s $$

Raw material cost:

$$ RawMaterialCost = \sum_{s\in S^R}\sum_{i:(i,s)\in ICS}\sum_{n\in N} C_s^{raw}\left(-\rho^{cons}_{i,s}\right) b_{i,n} $$

Discard cost:

$$ DiscardCost = c^{discard} \sum_{s \in S^I} \sum_{n \in N} Discard_{s,n} $$

The economic profit groups all market-facing terms:

$$ Profit = Revenue + GrapeSkinRevenue - OutsourcingCost - RawMaterialCost - LatenessCost - DiscardCost $$

Cooling cost for exterior tanks is modeled as a MILP. For non-storage tasks ($i \in I^{nst}$), the model introduces:

$$ z_{i,j,n} = b_{i,j,n} \cdot W_{i,n} $$

with exact linearization (because $W_{i,n}$ is binary):

$$ 0 \le z_{i,j,n} \le b_{i,j,n} $$

$$ z_{i,j,n} \ge b_{i,j,n} - B^{max}_{i,j}(1 - W_{i,n}) $$

$$ z_{i,j,n} \le B^{max}_{i,j} W_{i,n} $$

The model validates at build time that $\beta_i = 0$ for all exterior-unit tasks (a `ValueError` is raised if any such task has $\beta_i \ne 0$). With $\beta_i = 0$, the bilinear term vanishes and the cooling cost is linear:

$$ CoolingCost = c^{cool} \sum_{\substack{(i,j) \in IJ \\ j \in J^{ext} \\ i \in I^{nst}}} \sum_{n \in N} \alpha_i \, z_{i,j,n} $$

Storage cooling associated with $Alm$ tasks ($i \in I^{st}$) is intentionally excluded from the objective in the current formulation.

**Degeneracy-breaking penalty.** A small $\varepsilon$ term pulls all task start times as early as possible, breaking the degeneracy of floating tasks (tasks that can shift freely within a feasible window without changing the economic objective):

$$ TimePullPenalty = \varepsilon \sum_{i \in I} \sum_{n \in N} Ts_{i,n}, \quad \varepsilon = 10^{-6} $$

The coefficient is small enough that it never alters the economic ranking of solutions.

Full profit-maximization objective:

$$ \max Z = GrapeSkinRevenue - OutsourcingCost - LatenessCost - RawMaterialCost - DiscardCost - c^{MS} \cdot MS - CoolingCost - TimePullPenalty $$

### Barrique and Jar Pool Aggregation

Barrique and jar vessels are pooled rather than tracked individually. Instead of one binary $y_{i,j,n}$ per vessel, an integer count variable replaces the entire pool.

**Additional sets:**

- $I^{Bar} \subseteq I$: Barrique aging tasks
- $I^{Jar} \subseteq I$: Jar aging tasks

**Additional parameters:**

- $\bar{n}_{bar}$: Total number of barrique vessels in the pool
- $\bar{n}_{jar}$: Total number of jar vessels in the pool
- $B_{bar}$: Fixed capacity of one barrique (L)
- $B_{jar}$: Fixed capacity of one jar (L)

**Additional variables:**

- $NumBarr_{i,n} \in \mathbb{Z}_{\ge 0}$: Number of barriques used by task $i$ at event $n$; $NumBarr_{i,n} \le \bar{n}_{bar}$
- $NumJar_{i,n} \in \mathbb{Z}_{\ge 0}$: Number of jars used by task $i$ at event $n$; $NumJar_{i,n} \le \bar{n}_{jar}$

Batch size for pool tasks is determined by the vessel count and fixed capacity:

$$b_{i,n} = NumBarr_{i,n} \cdot B_{bar} \quad \forall i \in I^{Bar}$$

$$b_{i,n} = NumJar_{i,n} \cdot B_{jar} \quad \forall i \in I^{Jar}$$

**Activation-linked vessel-count bounds.** When a pool task is active, vessel counts are bounded by per-task minimum units and by pool size:

$$NumBarr_{i,n} \ge j^{min}_i\,W_{i,n} \quad \forall i \in I^{Bar}$$

$$NumJar_{i,n} \ge j^{min}_i\,W_{i,n} \quad \forall i \in I^{Jar}$$

$$NumBarr_{i,n} \le \bar{n}_{bar}\,W_{i,n} \quad \forall i \in I^{Bar}$$

$$NumJar_{i,n} \le \bar{n}_{jar}\,W_{i,n} \quad \forall i \in I^{Jar}$$

**Per-event-index pool capacity.** At each event point, the total vessels in use across all pool tasks cannot exceed the pool size:

$$\sum_{i \in I^{Bar}} NumBarr_{i,n} \le \bar{n}_{bar} \quad \forall n \in N$$

$$\sum_{i \in I^{Jar}} NumJar_{i,n} \le \bar{n}_{jar} \quad \forall n \in N$$

This handles within-event conflicts. Cross-event conflicts (tasks at different event indices that overlap in time) are handled by the cross-event sequencing formulation below.

**Cross-event sequencing.** The per-vessel timing constraint A09 ($Tsj_{j,n+1} \ge Tfj_{j,n}$) and the pool capacity constraint have no direct equivalent when vessels are not tracked individually. They are replaced by a cross-event sequencing formulation.

After static preprocessing (event-range fixing), let $\mathcal{P}^{Bar}$ be the set of unordered pairs of barrique task-events that are not provably inactive:

$$\mathcal{P}^{Bar} = \bigl\{\{(i_1,n_1),(i_2,n_2)\} : i_1,i_2 \in I^{Bar},\ (i_1,n_1)\ne(i_2,n_2),\ \text{neither fixed inactive}\bigr\}$$

For each pair, two sequencing binaries encode the temporal ordering:

- $z^{fwd}_{i_1,n_1,i_2,n_2} \in \{0,1\}$: 1 if task-event $(i_1,n_1)$ finishes before $(i_2,n_2)$ starts
- $z^{rev}_{i_1,n_1,i_2,n_2} \in \{0,1\}$: 1 if task-event $(i_2,n_2)$ finishes before $(i_1,n_1)$ starts

When both are 0, the two task-events are concurrent.

**Forward sequencing:**

$$Tf_{i_1,n_1} \le Ts_{i_2,n_2} + M^{\ge}_{i_1,i_2}\bigl(1 - z^{fwd}_{i_1,n_1,i_2,n_2}\bigr)$$

**Reverse sequencing:**

$$Tf_{i_2,n_2} \le Ts_{i_1,n_1} + M^{\ge}_{i_2,i_1}\bigl(1 - z^{rev}_{i_1,n_1,i_2,n_2}\bigr)$$

**Pool capacity (cross-event overlap coupling; complements per-event bounds and replaces per-vessel A09 for pooled units):**

$$NumBarr_{i_1,n_1} + NumBarr_{i_2,n_2} \le \bar{n}_{bar} + \bar{n}_{bar}\bigl(z^{fwd}_{i_1,n_1,i_2,n_2} + z^{rev}_{i_1,n_1,i_2,n_2}\bigr)$$

When both sequencing binaries are 0 (concurrent), this reduces to $NumBarr_{i_1,n_1} + NumBarr_{i_2,n_2} \le \bar{n}_{bar}$. When either is 1 (sequential), the right-hand side is relaxed to $2\bar{n}_{bar}$, so no additional cross-event coupling is needed for that pair. The per-event bounds $\sum_i NumBarr_{i,n} \le \bar{n}_{bar}$ and $\sum_i NumJar_{i,n} \le \bar{n}_{jar}$ remain active.

The same formulation applies to $\mathcal{P}^{Jar}$ with $NumJar$, $\bar{n}_{jar}$, $z^{fwd/rev}$ replaced by their jar counterparts.

**Maximum idle time (BarriqueMaxIdle).** Barriques must be refilled within 24 hours of being emptied. When $(i_1,n_1)$ precedes $(i_2,n_2)$:

$$Ts_{i_2,n_2} \le Tf_{i_1,n_1} + 24 + M^{max}_{i_1,i_2}\bigl(1 - z^{fwd}_{i_1,n_1,i_2,n_2}\bigr)$$

$$Ts_{i_1,n_1} \le Tf_{i_2,n_2} + 24 + M^{max}_{i_2,i_1}\bigl(1 - z^{rev}_{i_1,n_1,i_2,n_2}\bigr)$$

Since all vessels in a task-event share a single start time $Ts_{i,n}$, this constraint on the task-event level is equivalent to the per-vessel constraint: any vessel reused between two sequential task-events must be refilled within 24 hours, and because all vessels in the second task start at $Ts_{i_2,n_2}$ simultaneously, the constraint cannot be relaxed for a subset of vessels.
