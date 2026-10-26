# DPSynth Binaries (`bin/`)

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This directory houses the high-level entry points for interacting with the
DPSynth project. These scripts provide interfaces for generating synthetic data,
extracting domain metadata, and evaluating synthesis quality.

## High-Level Execution Flows

The project supports two primary paths for data generation depending on the
scale of the dataset:

### 1. Simple Local Generation (`main.py`)

Used for rapid prototyping or processing small datasets that fit into memory. It
leverages Pandas for I/O and interfaces directly with the local mechanisms in
`discrete_mechanisms/`.

**Example Usage:**
```bash
python3 bin/main.py \
  --dataset=/path/to/data.csv \
  --domain=/path/to/domain.yaml \
  --epsilon=1.0 \
  --delta=1e-8 \
  --mechanism=mst \
  --output_path=/path/to/output.csv
```

### 2. Distributed Production Generation (`run_data_generation.py`)

The primary binary for large-scale datasets. It orchestrates the lifecycle of
the [DatasetDescriptor](../dataset_descriptors/README.md), from initialization
(via format-specific generators) to metadata population (using
[pipeline_transformations](../pipeline_transformations/README.md)) and final
distributed sampling.
*   Supports various formats including CSV and TFRecord.
*   Can run locally (`--use_beam=false`) or as a distributed Dataflow job.

## Metadata Extraction (\"The Population Phase\")

### `derive_domain.py`

A standalone binary used to extract the initial schema and range metadata from a
dataset. It generates a YAML file compatible with [domain.py](../domain.py),
which can then be passed to `main.py` or used to bootstrap descriptors in more
complex pipelines.

## Evaluation & Monitoring

### `comparison.py`

Provides high-level diagnostic tools to evaluate the performance of the DP
synthesis. It generates histograms and utilizes SDMetrics to produce quality
reports comparing the real and synthetic distributions.

## Internal Helper Modules

These modules consolidate shared logic and flags for the binaries above:

*   **`_proto_class_flag.py`**: Shared Abseil flags for protobuf message type
    specification.
*   **`_read_csv_args.py`**: Standardized flags for CSV parsing (separators,
    field names).
