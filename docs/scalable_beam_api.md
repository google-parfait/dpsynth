# Scalable Beam API Guide

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

When your dataset is large enough to be processed on one machine (>1M records)
or when your need distributed processing, use the **Scalable Beam API**.

This interface is built on top of `pipeline_dp.PipelineBackend`, allowing the
mathematical synthesis transformations
in [`pipeline_transformations/`](../pipeline_transformations/README.md)
to run seamlessly on distributed computation engines
like **Apache Beam**
(or Apache Spark in open-source deployments).

--------------------------------------------------------------------------------

## Python API: `data_generation.generate`

The core entry point for distributed pipeline execution is `generate()` located
in [`dpsynth.data_generation`](../data_generation.py).
It operates directly on collections (such as Apache Beam `PCollection`s or
standard Python iterables) and returns a distributed collection of synthetic
tuples or records.

--------------------------------------------------------------------------------

## Creating a `DatasetDescriptor`

Before configuring the data generation job, you must build a `DatasetDescriptor`
corresponding to your input data format. DPSynth provides dedicated builder
utilities under
the [`dataset_descriptors/`](../dataset_descriptors/README.md)
module:

### 1. For CSV Datasets (`csv_descriptor.py`)

To build a descriptor from a CSV file, load a small representative sample (e.g.
first 100 rows) into a Pandas DataFrame. DPSynth will automatically deduce the
data type (`INT`, `STR`, `FLOAT`, `BOOL`) for each column.

```python
from dpsynth.dataset_descriptors import csv_descriptor

# Load sample from CSV file to deduce schema types
sample_df = csv_descriptor.read_csv_sample("/path/to/my_dataset.csv")

# Deduces types and instantiates AttributeDescriptors automatically
descriptor = csv_descriptor.get_dataset_descriptor_for_csv(
    dataframe=sample_df,
    field_names=["age", "occupation", "capital_gain"] # Optional: restrict columns to synthesize
)
```

### 3. For TFRecord Datasets (`tfrecord_descriptor.py`)

To build a descriptor from `tf.train.Example` records inside TFRecord files,
parse a representative sample list of records.

```python
from dpsynth.dataset_descriptors import tfrecord_descriptor

# Load first 1000 sample tf.train.Example records
sample_records = tfrecord_descriptor.read_tfrecords_sample(
    path="/path/to/sharded_data*.tfrecord",
    sample_size=1000
)

# Deduces types and validates that all sample records conform to same schema
descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
    sample_records=sample_records,
    attributes=["age", "label"] # Optional: filter attributes
)
```

--------------------------------------------------------------------------------

### Configuration: `DataGenerationConfig`

Before launching a job, instantiate a `DataGenerationConfig` dataclass to define
your privacy parameters, input/output formats, and mechanism preferences.

```python
from dpsynth import data_generation
from dpsynth.pipeline_transformations import types

config = data_generation.DataGenerationConfig(
    epsilon=1.0,
    delta=1e-7,
    mechanism=data_generation.Mechanism.MST, # or AIM, SWIFT, INDEPENDENT
    dataset_descriptor=descriptor_object,    # Schema descriptor (CSV, TFRecord)
    data_format=types.DataFormat.CSV,  # Input format
    output_format=types.DataFormat.CSV, # Output format (defaults to data_format)
    num_out_records=100_000,                 # Number of synthetic records to sample
)
```

### Running the Pipeline in Python

```python
import apache_beam as beam
from dpsynth import data_generation
import pipeline_dp

# 1. Instantiate your Beam pipeline
with beam.Pipeline() as pipeline:
    # 2. Load raw distributed data into a PCollection
    raw_records = pipeline | "ReadRecords" >> beam.io.ReadFromTFRecord("/path/to/data.tfrecord*")

    # 3. Instantiate BeamBackend
    beam_backend = pipeline_dp.BeamBackend()

    # 4. Optional: Capture diagnostic accounting and L1 loss metadata
    additional_output = data_generation.AdditionalOutput()

    # 5. Execute distributed generation pipeline
    synthetic_records = data_generation.generate(
        input_data=raw_records,
        config=config,
        backend=beam_backend,
        additional_output=additional_output,
    )

    # 6. Write synthetic records to distributed sink
    synthetic_records | "WriteSynthetic" >> beam.io.WriteToTFRecord("/path/to/synthetic.tfrecord")
```

--------------------------------------------------------------------------------

## Runnable Example

See `examples/pipeline_tabular_synthesis.py` for a self-contained example script
that loads a CSV dataset and YAML domain specification, builds a
`DatasetDescriptor`, and runs `data_generation.generate` with either
`pipeline_dp.LocalBackend` or `pipeline_dp.BeamBackend`.

--------------------------------------------------------------------------------

## Specifying Attribute Domains

While DPSynth can dynamically deduce column schemas and categorical listings,
you can explicitly configure attribute types, categories, and bounds using a
YAML domain specification file.

### ⚠️ Critical Requirement for Numerical Columns

> [!IMPORTANT] **For now, you MUST use a `domain.yaml` file to specify
> boundaries for all `NumericalAttributes`.**
>
> Currently, if a numerical column is not defined in the domain file, DPSynth
> attempts to derive its `min_value` and `max_value` boundaries dynamically from
> the raw dataset. However, this dynamic derivation is currently a **non-private
> operation** (marked as a developmental `TODO`).
>
> To ensure your pipeline satisfies strict mathematical Differential Privacy
> guarantees, you **must** pre-specify the absolute `min_value` and `max_value`
> bounds for every numerical column in your domain specification. These bounds
> should be determined using public information or analytical heuristics without
> looking at the sensitive data.

### Example: Applying a Domain Specification in Python

```python
import dpsynth

domains = dpsynth.from_yaml("/path/to/my_public_domain.yaml")
if isinstance(domains, dpsynth.Schema):
    domains = domains.attributes
descriptor.update_from_domain_specification(domains)
```

--------------------------------------------------------------------------------

## Saving & Re-Using the Trained Model

Generating differentially private synthetic data involves computing noisy
marginals and fitting a complex graphical model (Private-PGM Markov Random
Field) over your sensitive data. This process consumes a portion of your privacy
budget (epsilon, delta) and can be computationally intensive for large datasets.

To solve this, DPSynth allows you to **save the trained model** (alongside its
mapping schema / descriptor) to a file. Once saved, you can reload this model to
generate/sample additional synthetic records on-demand.

### Why Save the Model?

*   **Zero Extra Privacy Cost**: Sampling records directly from a previously
    trained model does **not** touch the sensitive source data and consumes
    **exactly 0 additional privacy budget**, regardless of how many billions of
    records you generate.
*   **Ultra-Fast Sampling**: Reloading the graphical model and sampling records
    from it bypasses the costly DP measurement and Private-PGM training phases.

--------------------------------------------------------------------------------

### 1. How to Save the Model

DPSynth packs the Markov Random Field (`mbi.MarkovRandomField`) and the
`DatasetDescriptor` into a single unified `SyntheticModel` Protocol Buffer
representation defined
in [`synthetic_model.proto`](../pipeline_transformations/synthetic_model.proto).

If building a custom pipeline, use the unified helper functions
in [`input_output.py`](../pipeline_transformations/input_output.py)
to serialize and save the MRF and descriptor collections.

--------------------------------------------------------------------------------

### 2. How to Load and Sample from a Saved Model

Once saved, you can reload the model and use the mathematical sampling layer to
generate fresh synthetic records.

You can load and sample directly within your pipeline scripts, allowing you to
easily plug synthetic generators into downstream ETL steps.
