# Wine Scheduling Model Equations

## Makespan minimization

This is a continuous-time, event-based scheduling model (STN-style) for wine production.
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
- $S^{zw} \subset S$: Zero-wait states
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
- $M$: Big-M parameter

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
$$ Ts_{i,n+1} \ge Tf_{i',n} - M(1 - W_{i',n}) $$

- `i` at `n+1` must start after `i′` at `n` finishes
- Big-M deactivates the constraint when tasks are inactive

**Zero Wait (ZW) Constraint (g12):**
For states $s \in S^{zw}$, if $i$ consumes $s$ from $i'$:
$$ Ts_{i,n+1} \le Tf_{i',n} + M(2 - W_{i',n} - W_{i,n+1}) $$

Zero-wait: consumption must start exactly when production ends

**No Intermediate Storage (NIS) Constraint (g13):**
For states $s \in S^{nis}$, if $i$ consumes $s$ from $i'$:
$$ Ts_{i,n+1} \le Tf_{i',n} + M(2 - W_{i',n} - W_{i,n+1}) $$

No-intermediate-storage: material cannot sit in inventory

### 3. Unit Assignment and Capacity

Total time spent by all tasks on a unit cannot exceed the horizon H. This prevents hidden overlaps.

Each task:

- needs at least `jMin` units
- can use at most `jMax` units
- only if the task is active

This supports parallel units (e.g. multiple tanks for one blend).

**Unit Capacity Horizon (h09):**
$$ \sum_{i \in I^{nst}} \sum_{n} (\alpha_i y_{i,j,n} + \beta_i b_{i,j,n}) \le H $$

**Max Units per Task (A01, A01st):**
$$ \sum_{j} y_{i,j,n} \le jMax \cdot W_{i,n} $$

**Min Units per Task (A02, A02st):**
$$ \sum_{j} y_{i,j,n} \ge jMin \cdot W_{i,n} $$

**Single Task per Unit (A03):**
$$ \sum_{i} y_{i,j,n} \le 1 $$

Each unit can handle at most one task per event.

### 4. Batch Sizing

**Batch Aggregation (A04):**
$$ b_{i,n} = \sum_{j} b_{i,j,n} $$

Total batch size equals the sum over assigned units.

**Batch Limits (A05a, A05b):**
$$ B_{i,j}^{min} y_{i,j,n} \le b_{i,j,n} \le B_{i,j}^{max} y_{i,j,n} $$

If a unit is used, batch size must lie between min and max.
If it is not used, batch size is forced to zero.

### 5. Operational Constraints

**Max Activations - Non-storage tasks (A06):**
$$ \sum_{n} W_{i,n} \le iMax \quad \forall i \in I^{nst} $$

Limits how many times a non-storage task can occur (e.g. one fermentation per batch).

**Max Activations - Storage tasks (A06st):**
$$ W_{i,N} \le iMaxST \quad \forall i \in I^{st} $$

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

**Ecobulk Constraints (A16, A17):**

**Ecobulk Timing (A16):**
For states $s \in S^{FISEco}$ (Ecobulk compatible), if task $i$ consumes $s$ produced by task $i'$:
$$ Ts_{i,n+1} \le Tf_{i',n} + M(2 - W_{i',n} - W_{i,n+1}) $$

This enforces a "no intermediate storage" condition for Ecobulk products. The consuming task must start immediately after the producing task finishes (or before, effectively zero-wait when combined with precedence).

**Mandatory Deposit Usage (A17):**
If $MustUseDep = 1$, for each state $s \in S^{FISEco}$:
$$ \sum_{n} \sum_{i \in ICS_s} W_{i,n} \ge 1 $$

This constraint forces the system to use at least one storage task (deposit) for Ecobulk-compatible products if the parameter is set, preventing the model from bypassing storage completely.

### 8. Final State and Objective

**Demand Satisfaction (g17):**
$$ ProdFinal_s \ge D_s $$

Final production must meet or exceed demand.

**Final Production Calculation (A18):**
$$ ST_{s,N} + \sum_{i \in IPS_s} \rho_{i,s}^{prod} b_{i,N} = ProdFinal_s $$

Final inventory plus last-event production equals total produced.

**Makespan (A21):**
$$ Tf_{i,N} \le MS \quad \forall i \in I^{pst} $$

The finish time of all pre-storage tasks must be ≤ MS.

**Objective Function:**
$$ \min Z = MS + Penalty_{unused} - Penalty_{space} $$

Minimize makespan while trying to:

- penalize unused capacity
- reward efficient storage usage
