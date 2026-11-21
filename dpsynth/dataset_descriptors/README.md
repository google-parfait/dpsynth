# Dataset Descriptors

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This directory provides the core implementation for describing structured
datasets and converting them to uniform formats suitable for Differential
Privacy (DP) tabular data synthesis algorithms.

`DatasetDescriptor` serves as the **central orchestration object** (the "glue")
of the DPSynth project, bridging format-specific data loading with core privacy
mechanisms.

## Project Architecture & Role

The `dataset_descriptors` package acts as an abstraction layer between raw data
sources and the internal mathematical representations used for DP synthesis. It
interacts with several other key modules:

*   **[domain.py](../domain.py)**: `AttributeDescriptor` relies on
    `CategoricalAttribute` and `NumericalAttribute` from `domain.py` to define
    the support and constraints of individual columns.
*   **[transformations.py](../transformations.py)**: `DatasetDescriptor` uses
    these tools to `encode()` raw values into discrete indices and `decode()`
    them back to the original domain.
*   **[common.py](../statistical/base/common.py)**: Compression logic
    (mapping rare values to an "Other" category) utilizes utilities from this
    module, often storing the results as `mbi.LinearMeasurement` objects within
    the descriptors.
*   **[pipeline_transformations/](../pipeline_transformations/)**: Provides the
    DP mechanisms that populate a descriptor's missing metadata (like quantiles
    or frequent categories) at scale using Apache Beam.

## Core Concepts

*   **`DatasetDescriptor`**: Holds the aggregated structure of a dataset. It
    encapsulates a list of `AttributeDescriptor`s alongside a format-specific
    `DataRecordConverter`. It provides the high-level API to `encode`, `decode`,
    `compress`, and `uncompress` dataset records.
*   **`AttributeDescriptor`**: Describes an individual dataset attribute (e.g.,
    a SQL column or Proto field). It manages the attribute's `DataType`, its
    domain (via `domain.py`), and any privacy-related metadata like quantiles or
    one-way marginal measurements.
*   **`DataRecordConverter`**: An abstract interface that bridges
    format-specific entities (CSV rows, Protos, TFRecords) into universal Python
    tuples used internally by the pipeline.
*   **`DataType` Enum**: Defines the fundamental types supported: `INT`, `ENUM`,
    `BOOL`, `STR`, and `FLOAT`.

## Workflow & Lifecycle

A `DatasetDescriptor` typically moves through three phases in a synthesis
pipeline:

1.  **Initialization**: Created from a schema or sample (e.g., via
    `get_dataset_descriptor_for_proto`). At this stage, it knows the names and
    types of attributes but might lack detailed domain info (like exact bounds).
2.  **Population (DP Derivation)**: In a DP pipeline, mechanisms (found in
    `pipeline_transformations/`) securely derive domain metadata—such as
    numerical quantiles or categorical value lists—and update the descriptor.
3.  **Encoding**: The fully populated descriptor is used by `data_generation.py`
    to transform raw input data into encoded discrete tensors for DP mechanisms
    like AIM or MST.
4.  **Decoding**: The descriptor is used to decode the synthetic results back
    into the original format.

## Supported Formats

The package provides specialized generators for primary dataset structures:

*   **CSV (`csv_descriptor.py`)**: Uses `get_dataset_descriptor_for_csv`.
    Deduces types from a Pandas DataFrame sample and produces a `CSVConverter`.
*   **Protocol Buffers (`proto_descriptors.py`)**: Uses
    `get_dataset_descriptor_for_proto`. Recursively traverses `.proto` message
    definitions to build a native field graph and a `ProtoConverter`.
*   **TFRecord (`tfrecord_descriptor.py`)**: Uses
    `get_dataset_descriptor_for_tfrecord`. Scans `tf.train.Example` records to
    deduce structure and returns a descriptor with a `TFRecordConverter`.

## Build Configuration

-   Base descriptors and the `DataRecordConverter` interface reside within the
    `:dataset_descriptor` build target.
-   Format-specific descriptor generators (e.g., `:csv_descriptor` or
    `:proto_descriptors`) can be depended on individually to avoid linking
    unnecessary heavy dependencies (Pandas, TF, etc.).
