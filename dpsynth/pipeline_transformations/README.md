# Pipeline Transformations for Synthetic Tabular Data

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This directory contains pipeline transformations which are used for
Differentially Private (DP) synthetic tabular data generation. These
transformations are agnostic to a pipeline framework, e.g., they can be run with
Apache Beam. Independence from frameworks is achieved by using
[`pipeline_dp.PipelineBackend`](https://github.com/google/pipeline-dp).
<!-- copybara:replace(b/491730365) -->
<!-- [`pipeline_dp.PipelineBackend`](https://github.com/OpenMined/PipelineDP). -->

## Project Role & Integration

`pipeline_transformations` serves as the distributed execution engine of the
project. It interacts deeply with other core modules:

*   **[dataset_descriptors/](../dataset_descriptors/)**: The transformations
    here are the primary engine for the **Population Phase** of the
    `DatasetDescriptor`. They securely derive missing metadata (like bounds or
    categories) from raw data.
*   **[statistical/base/](../statistical/base/)**: While this directory
    implements Beam-native versions of mechanisms, it frequently reuses the
    shared mathematical utilities (e.g., domain compression logic) found in
    `statistical/base/common.py`.

## Functional Overview

Modules are organized based on their stage in the synthesis pipeline:

### 1. Data I/O & Foundation

*   **`input_output.py`**: Bridges raw storage (CSV, TFRecord
    )
    with the pipeline.
*   **`types.py`**: Standard type aliases for PCollections and internal data
    structures.
*   **`diagnostic_info.py`**: Tracks execution metadata, DP accounting, and
    utility metrics (e.g., L1 distances) at scale.

### 2. Domain Metadata Derivation (Population Phase)

These transformations securely fill an initially "uninitialized"
`DatasetDescriptor` with real-world data bounds.

*   **`categorical_values_derivation.py`**: Derives categorical value lists
    using DP partition selection.
*   **`numerical_values_derivation.py` & `dp_auto_discretizer.py`**: Computes DP
    percentiles to automatically define numerical bins.

### 3. Data Transformation & Compression

*   **`dataset_encoding.py`**: Orchestrates the mapping from raw records to
    discrete integer indices.
*   **`dataset_compression.py` & `compression.py`**: Merges rare values into an
    "Other" category based on DP one-way marginals, reducing the state space for
    mechanisms.

### 4. Core DP Building Blocks

*   **`marginals_computations.py`**: The low-level interface to `pipeline_dp`
    and `mbi`. Computes noised marginals and structures them as
    `LinearMeasurement` objects.

### 5. Modeling & Generation

*   **`model.py`**: Fits the distributed global model (using `mbi` iterative
    fitting) and samples the final synthetic records in parallel batches.

### 6. Distributed DP Mechanisms

Orchestrate the above components into end-to-end algorithms:

*   **`aim.py`**: Distributed Adaptive+Iterative Mechanism.
*   **`mst.py`**: Distributed Maximum Spanning Tree Mechanism.
*   **`independent_mechanism.py`**: Baseline independent attribute generation.

## Relationship to `statistical/base`

While `statistical/base` is designed for **local, single-machine**
prototyping on small datasets (using Pandas/NumPy), `pipeline_transformations`
is built for massive-scale production datasets. In many cases, the pipeline
implementation calls into the same core mathematical functions after the initial
distributed data reduction.
