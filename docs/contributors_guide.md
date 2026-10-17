# Contributor Guide

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

Welcome to the **DPSynth** contributor guide! This document is written for
current and future developers, maintainers, and contributors. It outlines the
codebase architecture, core abstractions, development guardrails for distributed
pipeline programming, diagnostic tracking, and the tabular evaluation framework.

--------------------------------------------------------------------------------

## Codebase Architecture & Directory Structure

DPSynth is architected with a strict boundary separating local mathematical DP
mechanisms from distributed execution transformations.

```
dpsynth/
 ├── bin/                         <-- Executable binaries (`main.py`, `run_data_generation.py`, `run_tabular_eval.py`)
 ├── data_generation.py           <-- Core Library API for distributed generation
 ├── dataset_descriptors/         <-- Format descriptors & schema converters (CSV, TFRecord)
 ├── discrete_mechanisms/         <-- Core mechanism implementations defined over integer-coded data represented as numpy arrays.  Also works with pre-computed marginal histograms instead of raw data
 ├── pipeline_transformations/    <-- Distributed pipeline implementations of DP transformations
 ├── eval/                        <-- Tabular evaluation engine & metrics (TV Distance, Cramer's V)
 └── domain.py                    <-- Public data model & attribute domain definitions
```

### Architectural Separation: General vs. Distributed

*   **`discrete_mechanisms/` (General Abstractions)**: Implements core
    mathematical algorithms (`AIM`, `MST`, `SWIFT`, `INDEPENDENT`). Operating
    strictly on discrete integer data (`mbi.Dataset`), these modules are
    single-machine, in-memory engines. They contain zero references to
    distributed frameworks
    like Apache Beam.
*   **`pipeline_transformations/` (Distributed Abstractions)**: Implements
    massive-scale, parallelized pipeline operations (e.g., distributed quantile
    bucketing, one-way marginal measurement, and batch record sampling). These
    modules orchestrate data preparation at scale before calling into the shared
    mathematical routines in `discrete_mechanisms/common.py`.

--------------------------------------------------------------------------------

## Core Orchestration Abstractions

The central abstraction bridging data formats with mathematical mechanisms is
the **DatasetDescriptor** located
in [`dataset_descriptors/dataset_descriptor.py`](../dataset_descriptors/dataset_descriptor.py).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DatasetDescriptor                             │
│                                                                         │
│  ┌─────────────────────────┐           ┌─────────────────────────────┐  │
│  │  AttributeDescriptors   │           │     DataRecordConverter     │  │
│  │  • age (Numerical)      │           │     • CSVConverter          │  │
│  │  • state (Categorical)  │ <───────> │     • ProtoConverter        │  │
│  │  • city (Open-Set)      │           │     • TFRecordConverter     │  │
│  └─────────────────────────┘           └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

*   **`DatasetDescriptor`**: Holds the aggregated schema of a dataset. It
    encapsulates a list of `AttributeDescriptor`s alongside a format-specific
    `DataRecordConverter`. It exposes the top-level API to `encode()`,
    `decode()`, `compress()`, and `uncompress()` records during the
    [Processing Lifecycle](processing_lifecycle.md).
*   **`AttributeDescriptor`**: Describes an individual column. It manages the
    attribute's data type (`INT`, `STR`, `FLOAT`, `BOOL`, `ENUM`), its
    underlying domain bounds, and private metadata like derived quantiles or
    compressed rare-category mappings.
*   **`DataRecordConverter`**: An interface bridging format-specific entities
    (CSV rows, Protobuf messages, TFRecord tensors) into standardized Python
    tuples used internally across pipeline collections.

--------------------------------------------------------------------------------

## Guide for Writing Pipelines with `PipelineBackend`

DPSynth enforces a strict development standard: all distributed data processing
must be written against the `pipeline_dp.PipelineBackend` abstraction rather
than any concrete execution framework.

**Core Interface
File**: [`pipeline_backend.py`](https://github.com/OpenMined/PipelineDP/blob/main/pipeline_dp/pipeline_backend.py)

### 1. What is `PipelineBackend`?

`PipelineBackend` is a polymorphic wrapper interface. It does not implement any
data manipulation itself; instead, it exposes unified signatures (like `map`,
`filter`, `group_by_key`) and delegates the actual operations to the native
execution engine in its concrete subclasses.

This design abstracts away the execution environment completely:

*   **Local Execution (`LocalBackend`)**: Wraps standard Python generators and
    iterables.

    *   *How `map`
        works*: [`LocalBackend.map()`](https://github.com/OpenMined/PipelineDP/blob/main/pipeline_dp/local_backend.py)
        applies the map function lazily using basic Python comprehensions:

    ```python
    def map(self, col, fn, stage_name):
        return (fn(x) for x in col)
    ```

*   **Distributed Execution (`BeamBackend`)**: Wraps Apache Beam's pipelines and
    collections (`PCollection`).

    *   *How `map`
        works*: [`BeamBackend.map()`](https://github.com/OpenMined/PipelineDP/blob/main/pipeline_dp/beam_backend.py)
        translates the wrapper signature into a native Beam transformation pipe:

    ```python
    def map(self, col, fn, stage_name):
        return col | stage_name >> beam.Map(fn)
    ```

### 2. Why Framework-Agnostic?

By writing all your data generation stages against this backend wrapper, the
exact same python pipeline function can be run locally during unit tests
(instant startup) or submitted to massive distributed clusters
without changing a single line of code.

*   **`pipeline_dp.LocalBackend`**: Executes locally. Essential for rapid unit
    testing and debugging.
*   **`pipeline_dp.BeamBackend`**: Executes
    on Apache Beam.
    In open-source deployments, it allows DPSynth to run on Apache Spark
    or Google Cloud Dataflow.

### 3. Designing Pipeline Functions

When writing new transformations in `pipeline_transformations/`, functions
**MUST** accept a `backend: PipelineBackend` instance as an argument and use its
methods exclusively for all data transformations.

```python
from pipeline_dp import PipelineBackend

def compute_custom_column_stats(backend: PipelineBackend, input_collection):
    # input_collection is framework-agnostic (PCollection or Python list)
    # Every operation MUST include a clear, descriptive stage_name

    # 1. Map: Extract column value
    col_values = backend.map(input_collection, lambda row: row[2], "ExtractColumnValue")

    # 2. Filter: Remove missing entries
    filtered_values = backend.filter(col_values, lambda x: x is not None, "FilterNonNull")

    # 3. Key: Pair with constant key
    keyed_data = backend.map(filtered_values, lambda x: ("all_records", x), "AddKey")

    # 4. Group & Reduce: Sum values per key
    grouped_data = backend.group_by_key(keyed_data, "GroupByKey")
    summed_data = backend.map_values(grouped_data, sum, "SumValues")

    return summed_data
```

### 3. Core `PipelineBackend` Methods Available

*   `map(col, fn, stage_name)`: Apply a function to each element.
*   `flat_map(col, fn, stage_name)`: Apply a function returning an iterable,
    flattening results.
*   `filter(col, fn, stage_name)`: Keep elements where `fn` returns `True`.
*   `group_by_key(col, stage_name)`: Group key-value pairs by key.
*   `map_values(col, fn, stage_name)`: Apply a function to values of key-value
    pairs.
*   `sum_per_key(col, stage_name)` / `reduce_per_key(col, fn, stage_name)`:
    Aggregate values per key.
*   `flatten(cols, stage_name)`: Merge multiple collections into one.
*   `distinct(col, stage_name)`: Get unique elements.

### 4. Critical Constraints & Safety Guardrails

> [!CAUTION] **Never use framework-specific operations** like `beam.Map`,
> `beam.Filter`, or the Beam pipe operator (`|`) inside functions designed to be
> backend-agnostic. All transformations must route through methods of the passed
> `backend` object.
>
> **Never assume collection types**. `input_collection` will be a Python
> iterable for `LocalBackend` but a `PCollection` for `BeamBackend`. Pipeline
> logic must never invoke methods specific to `list` or `PCollection`.

### 5. PipelineDP Integration & DP Aggregations

While the `PipelineBackend` abstraction is used to write the framework-agnostic
data manipulation pipeline, the actual **Differentially Private Aggregations**
are powered by the `pipeline_dp.DPEngine` object from
the [PipelineDP](https://github.com/OpenMined/PipelineDP)
library.

When contributing new mechanisms or data transformations, verify that you
leverage the correct PipelineDP primitives for privacy-preserving operations:

*   **DP Count (Noisy Marginals)**: Used to compute noisy categorical value
    counts (univariate and multi-dimensional joint frequencies). This is
    executed via the `DPEngine.aggregate()` layer in `marginals_computations.py`
    and during domain compression.
*   **DP Quantiles (Discretization Bounds)**: Used to privately calculate
    equal-frequency boundaries for binning continuous numerical attributes. This
    is handled via `DPEngine.aggregate()` using `Percentile` metrics in
    `numerical_values_derivation.py`.
*   **DP Partition Selection (Attribute Discovery)**: Used to discover frequent
    categories in raw open-set text data while safely filtering out highly
    identifying rare categories. This is implemented via
    `DPEngine.select_partitions()` in `categorical_values_derivation.py`.

--------------------------------------------------------------------------------

## Diagnostic Information & DP Accounting

When executing distributed jobs, tracking differential privacy budget
consumption and synthesis fidelity is vital. DPSynth captures this metadata in a
standardized Protocol Buffer defined
in [`diagnostic_info.proto`](../diagnostic_info.proto).

```protobuf
message DiagnosticInformation {
  optional double epsilon = 1;
  optional double delta = 2;
  optional string mechanism = 3;
  repeated string attribute_names = 4;
  repeated int64 compressed_attribute_sizes = 5;
  repeated DPOperation dp_operations = 6;
}
```

When launching jobs via `bin/run_data_generation.py`, passing
`--diagnostic_information_path` outputs this proto. It tracks:

*   **`compressed_attribute_sizes`**: The exact domain state space size after
    rare-category compression.
*   **`dp_operations`**: A breakdown of every individual DP noise addition
    (e.g., quantile derivations, 1-way marginals, 2-way spanning tree
    measurements), listing the exact `epsilon`, `delta`, and noise standard
    deviation (`sigma`) consumed.

--------------------------------------------------------------------------------

## Marginal-Based Inference (`mbi`)

The core mathematical optimization, graphical model fitting, and sampling logic
in DPSynth is powered by the
third-party [Marginal-Based Inference (mbi)](https://github.com/ryan112358/mbi)
library. All discrete DP mechanisms (`AIM`, `MST`, `SWIFT`, `INDEPENDENT`)
leverage the datastructures and solvers inside this package.

### 1. Domain & Data Abstractions

`mbi` operates strictly over finite discrete integer spaces. It provides the
following core classes:

*   **`mbi.Domain`**: Defines the joint discrete dimensions of all columns. It
    maps attribute keys to their discrete size (`0 ... K-1`). Core code resides
    in [`domain.py`](https://github.com/ryan112358/mbi/blob/master/mbi/domain.py).
*   **`mbi.Dataset`**: Represents a discretized dataset. Internally, it wraps a
    Pandas `DataFrame` where every row is encoded as a dense vector of
    zero-indexed integers conforming to the `mbi.Domain`. Core code resides
    in [`dataset.py`](https://github.com/ryan112358/mbi/blob/master/mbi/dataset.py).
*   **`mbi.LinearMeasurement`**: Holds a noisy measured marginal query. It
    encapsulates a query matrix (defining which subset of columns is measured)
    alongside the differentially private categorical count vector. Core code
    resides
    in [`marginal_loss.py`](https://github.com/ryan112358/mbi/blob/master/mbi/marginal_loss.py).

### 2. Markov Random Field ([`mbi.MarkovRandomField`](https://github.com/ryan112358/mbi/blob/master/mbi/markov_random_field.py))

The probablistic model that represents the synthetic distribution is a **Markov
Random Field (MRF)**. * The MRF maintains potential parameters (cliques) over
small subsets of columns representing pairwise or low-dimensional attribute
correlations. * Rather than maintaining a massive joint distribution table, the
MRF factorizes the joint probability as a product of these low-dimensional
potentials.

### 3. Mirror Descent Optimization & Private-PGM Solver

The core PGM inference solver resides
in [`mbi.estimation.MirrorDescent`](https://github.com/ryan112358/mbi/blob/master/mbi/estimation.py).

* **The API**:

```python
  model = mbi.estimation.MirrorDescent(
      marginal_oracle=...,  # optional
  ).estimate(
      domain: mbi.Domain,
      linear_measurements: list[mbi.LinearMeasurement],
      iters: int = 2500
  )
```

*   **How it works**: It uses mirror descent (Multiplicative Weights) to
    optimize the clique potentials of the `MarkovRandomField`. The objective is
    to minimize the reconstruction error (such as L1 or L2 loss) between the
    model's projected marginal distributions and the measured noisy
    `LinearMeasurement` objects.
*   **Fidelity Enforcements**: Mirror descent rigorously ensures that the
    optimized joint distribution is mathematically consistent, globally
    normalized, and strictly non-negative across all overlapping and
    intersecting marginal measurements.
*   **Sampling**: The returned `MarkovRandomField` object acts as a generative
    sampler. Call `model.synthetic_data(rows)` to run parallelized randomized
    rounding over the junction tree and draw an arbitrary number of
    high-fidelity, consistent synthetic integer records.

--------------------------------------------------------------------------------

## The Tabular Evaluation Framework (`eval/`)

To rigorously evaluate the quality and statistical fidelity of generated
synthetic data compared to real sensitive data, DPSynth includes a comprehensive
evaluation engine located
in [`eval/`](../eval/BUILD)
and executable
via [`bin/run_tabular_eval.py`](../bin/run_tabular_eval.py).

```
┌────────────────────────────────────────────────────────┐
│ Sensitive Data (Real)   &   Synthetic Data (Generated) │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Tabular Evaluation Engine (`eval/tabular_eval.py`)     │
│                                                        │
│  1. Univariate Stats (Mean, Std, Min, Max, Nulls)      │
│  2. One-Way Marginal Distances (TV Distance, Chi-Sq)   │
│  3. Bivariate Correlations (Cramer's V Joint Counts)   │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│         TabularEvalReport Protobuf Dashboard           │
└────────────────────────────────────────────────────────┘
```

### Evaluation Metrics & Workflow

The evaluation framework is engine-agnostic (powered by `PipelineBackend`),
allowing it to run locally on small CSVs or distributed across clusters on
sharded
tables.
It computes:

1.  **Univariate Statistics**: Evaluates basic moments (mean, standard
    deviation, min, max, missing value counts) for numerical columns using
    distributed accumulators (`NumericalStatisticsCombiner`).
2.  **Marginal Distribution Distances**: Measures the statistical gap between
    real and synthetic 1-way marginal distributions using Total Variation (TV)
    distance and Chi-squared metrics.
3.  **Bivariate Correlations (Cramer's V)**: Evaluates cross-column relational
    integrity. It counts joint frequencies across categorical attribute pairs
    `(col_A, col_B)` in both datasets and derives Cramer's V correlation
    matrices to verify that multi-column dependencies were correctly preserved
    by graphical model inference.

### Running Evaluation (CLI)

```bash
python3 bin/run_tabular_eval.py \
  --original_data_path=/path/to/real_data.csv \
  --synthetic_data_path=/path/to/synthetic_data.csv \
  --eval_report_path=/tmp/eval_report.pb \
  --data_format=csv \
  --use_beam=false
```
