# Wine Scheduling Model Equations

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

Activation and unit-assignment limits:

- $i_{max}$: Maximum number of activations for non-storage tasks
- $j_{max}$: Maximum number of units assigned to an active non-storage task at an event
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

$$ \sum_{n} W_{i,n} \le i_{max} \quad \forall i \in I^{nst} $$

Limits how many times a non-storage task can occur (e.g. one fermentation per batch).

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

$$ \min Z = MS + Penalty_{unused} + Penalty_{space} $$

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
\forall j \in J_i: y_{i,j,n} = 0, \; b_{i,j,n} = 0
\end{bmatrix}$$

A global constraint enforces at most one task per unit per event:

$$ \sum_{i} y_{i,j,n} \le 1 \quad \forall j, n $$

Duration in the active disjunct differs by task type:

$$\text{Non-storage } (i \in I^{nst}): \quad Tf_{i,n} = Ts_{i,n} + \alpha_i + \beta_i b_{i,n}$$

Storage tasks do not use a fixed duration equation in the active disjunct. Their behavior is modeled by persistence implications ($W_{i,n} \implies W_{i,n+1}$) together with a terminal consistency condition embedded directly in the final active disjunct: if a storage task is active at the final event, then $Tf_{i,N} = MS$. This prevents a storage task that first activates at $N$ from taking zero duration while avoiding global Big-M coupling for this logic.

### 2. Global Precedence Logic (Eq. 2.2)

Instead of 3-way disjunctions, precedence is expressed as conditional constraints using Big-M implications on $W$.

**General Precedence:**

$$ W_{i',n} = 1 \land W_{i,n+1} = 1 \implies Ts_{i,n+1} \ge Tf_{i',n} $$

Linearized as:

$$ Ts_{i,n+1} \ge Tf_{i',n} - H(2 - W_{i',n} - W_{i,n+1}) $$

**Zero-Wait / No-Intermediate-Storage / Ecobulk (equality when both active):**

$$ W_{i',n} = 1 \land W_{i,n+1} = 1 \implies Ts_{i,n+1} = Tf_{i',n} $$

Linearized as the $\ge$ above plus:

$$ Ts_{i,n+1} \le Tf_{i',n} + H(2 - W_{i',n} - W_{i,n+1}) $$

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

**Global Capacity:**

$$ \sum_{n} W_{i,n} \le i_{max} $$

**Unit Constraints:**

$$ \sum_{j} y_{i,j,n} \le j_{max} \cdot W_{i,n} $$

### 5. Variable Definitions (GDP Adaptations)

- **$W_{i,n}$:** The GDP disjunct's `binary_indicator_var`, no separate binary variable is declared. This single binary serves both the disjunction structure and algebraic constraints (A01, A06, precedence, persistence, etc.), and activates terminal storage horizon sync inside the final storage disjunct.
- **$y_{i,j,n}$:** Unit assignment binary; retained for global constraints (A03, batch limits).
