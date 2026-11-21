# Discrete Mechanisms for Synthetic Tabular Data

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

Discrete mechanisms are the core in-memory synthesis algorithms in DPSynth. They
generate differentially private synthetic tabular data by learning a
probabilistic model of an already discretized dataset.

Rather than copying or perturbing individual records, a mechanism privately
measures aggregate statistics—called **marginals**—over one or more columns. It
then uses those noisy measurements to fit a graphical model
(`mbi.MarkovRandomField`) and samples new synthetic records from that model.

At a high level, every discrete mechanism follows this pattern:

```text
encoded sensitive data
    -> measure one-way marginals
    -> optionally compress rare values
    -> select useful additional marginals
    -> measure them with DP noise
    -> estimate a graphical model
    -> sample synthetic discrete records
```

## `base.py` — Shared execution pipeline

**Role:** Owns the common Select-Measure-Estimate lifecycle.

**Main logic:** Calibrates privacy budget, obtains initial one-way measurements,
compresses domains, runs selection/measurement/estimation, decompresses sampled
data, and returns `DiscreteMechanismResult`.

**Public API:** `DiscreteMechanism.configure()`

## `common.py` — Main Shared Utility Layer

This is the main utility module for the discrete-mechanism workflow. It provides
the reusable operations used across mechanisms: noisy marginal measurement,
workload and clique helpers, domain compression, timing, and diagnostics. It
also defines `DiscreteMechanismResult`, returned by every mechanism, and
`MechanismDiagnostics`. Add generic logic here only when it is shared by
multiple mechanisms; keep selection policy in individual mechanism files.

## `independent.py` - One-Way Baseline

Implements `IndependentConfig`, the simplest mechanism. It uses the shared
workflow from `base.py`, spends its budget on one-way marginals, and selects no
additional cliques. The resulting model preserves each column's distribution but
does not model relationships between columns.

**Public API:** `IndependentConfig`

## `direct.py` — Caller-Defined Workload

Implements `DirectConfig`, which measures the cliques supplied through
`prespecified_marginal_queries`. It performs no data-dependent selection and
does not create its own one-way measurements, so the full budget is available
for the specified workload. Initial measurements supplied by another layer are
included when fitting the final model.

**Public API:** `DirectConfig(prespecified_marginal_queries=...)`

## `mst.py` — Private Pairwise Spanning-Tree Selection

Implements `MSTConfig`, the default general-purpose mechanism for preserving
pairwise relationships. It begins with one-way marginals, privately selects
pairwise cliques forming a maximum spanning tree, measures those cliques, and
uses the shared base pipeline to estimate the final model.

**Public API:** `MSTConfig`

**Internal behavior:** `_allocate_budget()` splits remaining rho between private
selection and measurement; `_select()` calls the spanning-tree selection logic.
`dp_maximum_spanning_tree()` and `_select_two_way_marginal_queries()` implement
the private pairwise-selection step.

## `aim.py` — Adaptive Iterative Selection

Implements `AIMConfig`, an adaptive workload-based mechanism. Instead of
selecting cliques once, it repeatedly finds a marginal that the current model
approximates poorly, measures it, and updates the model.

**Public API:** `AIMConfig(workload=...)`

**Internal behavior:** `_one_way_cliques()` limits initial measurements to the
workload; `_allocate_budget()` reserves rho for the adaptive loop; `_run()`
replaces the standard base execution path. Helper functions filter valid
candidates and privately choose the worst-approximated marginal.

## `swift.py` — Workload and Clique-Tree Mechanism

Implements `SWIFT`, a workload-informed mechanism that selects
marginals while controlling clique-tree complexity. It uses a custom
junction-tree-aware estimation and sampling path rather than the standard
one-pass implementation in `base.py`.

**Public API:** `SWIFTConfig(workload=...)`

**Internal behavior:** `_allocate_budget()` splits rho between selection and
measurement; `_run()` compiles the workload, selects supported cliques, builds a
clique tree, measures selected marginals, and estimates the final model.
Module-level clique-tree construction and query-selection functions support this
workflow and are internal implementation details.

## `clique_tree.py` and `swift_utils.py` — SWIFT Support Logic

These files contain utilities used only by SWIFT. `clique_tree.py` builds and
updates clique-tree structures while preserving support for selected marginals.
`swift_utils.py` represents candidate subsets and chooses a feasible subset and
budget allocation under SWIFT's model-size constraints.

## `accounting.py` — Legacy Privacy Conversions

Contains zCDP/GDP conversion helpers still used by the current mechanisms. New
conversion logic is expected to migrate to the external `dp_accounting` API.
