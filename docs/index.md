# DPSynth: Differentially Private Synthetic Data Generation

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

DPSynth is a comprehensive library and toolset for generating **Differentially
Private (DP) synthetic data** across a wide range of data modalities (including
discrete, tabular, relational, nested, sequence, and free-form text data). Given
a sensitive dataset of records, DPSynth generates a synthetic version of the
dataset that preserves the multi-dimensional structure, relational
correlations, and statistical properties of the original data while rigorously
satisfying mathematical differential privacy (`(epsilon, delta)-DP`).

--------------------------------------------------------------------------------

## Why DPSynth?

In modern data science, sharing sensitive datasets for analytics, testing, or
model training presents significant privacy risks. Standard anonymization
techniques (like masking or `k`-anonymity) are notoriously vulnerable to linkage
attacks.

DPSynth solves this by providing a unified library of Differential Privacy
mechanisms spanning discrete and graphical model synthesis (including **AIM**,
**MST**, and **SWIFT** paired with Private-PGM inference), relational and
nested synthesis, and LLM-based text and mixed-modality generation. The
resulting synthetic records can be freely shared, analyzed, or published
without compromising the privacy of any individual present in the source data.

--------------------------------------------------------------------------------

## Core APIs and Execution Models

All mechanisms in DPSynth implement the lightweight
[`MechanismConfig` / `CalibratedMechanism` API](mechanism_api.md). While the
library supports a wide range of modalities, they are currently at different
levels of maturity: basic tabular and discrete data are the most mature and
well-tested, whereas newer modalities in experimental and research modules are
still being developed, stress-tested, and improved. We expect the overall APIs
(`MechanismConfig`, `CalibratedMechanism`) to remain mostly unchanged, and
future changes should mostly (though not guaranteed) impact internal
implementations rather than the public-facing APIs.

For tabular and discrete synthesis specifically, DPSynth provides two unified
execution models designed to scale from small local dataframes to massive
distributed datasets across computing clusters:

```
               ┌────────────────────────────────────────┐
               │            Sensitive Dataset           │
               └───────────────────┬────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ Choose Execution Environment │
                    └──────────────┬───────────────┘
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│ In-Memory DataFrame API │                    │  Scalable Pipeline API  │
├─────────────────────────┤                    ├─────────────────────────┤
│ • Single-machine RAM    │                    │ • Massive cluster scale │
│ • Pandas DataFrame I/O  │                    │ • Apache Beam / Spark   │
│ • Python scripts & CLI  │                    │ • Sharded files / SQL   │
└───────────┬─────────────┘                    └───────────┬─────────────┘
            └──────────────────────┬───────────────────────┘
                                   ▼
               ┌────────────────────────────────────────┐
               │   Differentially Private Synthetic Data │
               └────────────────────────────────────────┘
```

### 1. In-Memory DataFrame API (`dpsynth.TabularConfig`)

Optimized for rapid prototyping, research experimentation, and datasets that
easily fit within single-machine memory.

*   **Inputs/Outputs**: Pandas DataFrames.
*   **CLI
    Binary**: [`bin/main.py`](../bin/main.py).
*   **Documentation**: [In-Memory DataFrame API Guide](in_memory_api.md).

### 2. Scalable Pipeline API (`dpsynth.data_generation`)

Built for production-grade data pipelines operating on massive distributed
datasets. Written against the framework-agnostic `pipeline_dp.PipelineBackend`
interface.

*   **Execution**: Runs
    on Apache Beam
    (or Apache Spark in open-source deployments).
*   **Inputs/Outputs**: (Sharded) CSV and TFRecord.
*   **CLI
    Binary**: [`bin/run_data_generation.py`](../bin/run_data_generation.py). *   **Documentation**:
    [Scalable PipelineBackend API Guide](scalable_beam_api.md).

--------------------------------------------------------------------------------

## Documentation Sitemap & Navigation

Whether you are a software engineer looking to integrate synthetic data into
your workflow, a researcher investigating graphical model synthesis, or a
contributor expanding the library, explore the documentation below:

*   **[Documentation Sitemap](sitemap.md)**: Complete table of contents and
    layout of the DPSynth documentation.
*   **[Specifying Attribute Domains](specifying_domains.md)**: Upfront domain
    requirements, supported attribute types, the DPSynth domain language, and
    inferring domains from Pydantic, Protobuf, and Pandas schemas.
*   **[Data Model & Terminology](data_and_terminology.md)**: Attributes
    (categorical vs. numerical), schema deduction, and `domain.yaml`
    specifications.
*   **[In-Memory DataFrame API Guide](in_memory_api.md)**: Detailed guide to
    using the Pandas-based API and local CLI. *   **[Scalable PipelineBackend API Guide](scalable_beam_api.md)**: Guide for
    distributed data generation using Beam and cluster execution.
*   **[Processing & Synthesis Lifecycle](processing_lifecycle.md)**: Deep dive
    into the 5-stage mathematical lifecycle: Derivation, Encoding, Compression,
    PGM Modeling, and Decoding.
*   **[Contributor & Architecture Guide](contributors_guide.md)**: Architectural
    separation, core abstractions, PipelineBackend programming rules,
    diagnostics, and the Tabular Evaluation framework.
*   **[Mechanism API Architecture](mechanism_api.md)**: Design decisions,
    the 3-step pipeline, configure vs. calibrate semantics, and tight
    privacy accounting.

--------------------------------------------------------------------------------

## Supported Synthesis Algorithms

DPSynth supports the following cutting-edge DP mechanisms across both of its
APIs:

*   **AIM (Adaptive Iterative Mechanism)**: An MWEM-style algorithm that
    iteratively selects and measures low-dimensional marginals based on
    workload, privacy budget, and data characteristics
    ([arXiv:2201.12677](https://arxiv.org/abs/2201.12677)).
*   **MST (Maximum Spanning Tree)**: Computes an approximate maximum spanning
    tree over pairwise attribute correlations using the exponential mechanism to
    model bivariate dependencies privately
    ([arXiv:2108.04978](https://arxiv.org/abs/2108.04978)).
*   **SWIFT**: An advanced, highly optimized mechanism for scalable synthesis of
    complex categorical and numerical distributions.
*   **INDEPENDENT**: A robust baseline mechanism that measures 1-way marginals
    via Gaussian noise and models all attributes independently.

--------------------------------------------------------------------------------

## Contact

--------------------------------------------------------------------------------

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

specifying_domains
data_and_terminology
in_memory_api
scalable_pipeline_api
```

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

processing_lifecycle
contributors_guide
```

```{toctree}
:maxdepth: 2
:caption: Developer documentation:
:hidden:

mechanism_api
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

api_reference
```
