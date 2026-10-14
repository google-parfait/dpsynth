# DPSynth: Differentially Private Synthetic Data Generation

[![Documentation](https://readthedocs.org/projects/dpsynth/badge/?version=latest)](https://dpsynth.readthedocs.io/en/latest/)

DPSynth is a library housing mechanisms for differentially private (DP)
synthetic data generation across a wide range of data modalities. Given a
sensitive dataset, our library can generate a synthetic version of the dataset,
preserving the structure and statistical properties of the source data while
satisfying differential privacy.

📖 **[Full documentation on ReadTheDocs](https://dpsynth.readthedocs.io/en/latest/)**

> [!WARNING]
> **This library is under active development.** While we expect the core
> public-facing APIs (`MechanismConfig`, `CalibratedMechanism`) to remain
> mostly unchanged, you may still encounter bugs, rough edges, or evolving
> internals. Standard tabular data settings (categorical and numerical
> attributes, single-table schemas) are the most mature and well-tested and
> should work well out-of-the-box, while newer modalities in experimental and
> research modules are still being developed, stress-tested, and improved.
>
> We have a long roadmap of features we plan to add. In the meantime, we
> welcome early adopters to:
>
> *   **Try it out** on your own datasets and use cases.
> *   **Report issues** — bugs, confusing behavior, or sharp edges.
> *   **Benchmark it** against other DP synthetic data implementations.
> *   **Suggest features** that would be valuable for your workflows.
> *   **Contribute** — whether it's a GitHub issue, pull request, new mechanism,
>     bug fix, or added functionality, contributions are welcome!
>
> Your feedback directly shapes the library's direction. Thank you for your
> patience as we build toward a stable release!

## Scope and Design Philosophy

DPSynth is a Python library for **differentially private synthetic data
generation**. This scope is deliberately inclusive and does not mean just
"tabular" data or "text" data. Methods for **any data modality** are in
principle in scope for the library, and community contributions for new
modalities and mechanisms are welcome.

Unlike [JAX Privacy](https://github.com/google-deepmind/jax_privacy), which
focuses on lower-level DP-SGD ingredients, DPSynth implements a wide variety of
DP mechanisms that naturally compose and build on top of each other:

*   **Supported Modalities**:
    *   **Discrete**: Low-level discrete, integer-encoded datasets via
        marginal-based mechanisms
        ([`DiscreteConfig`](dpsynth/discrete_mechanisms/discrete.py),
        [`discrete_mechanisms/`](dpsynth/discrete_mechanisms/README.md)).
    *   **Tabular (Single-Table)**: Categorical, numerical, and open-set
        attributes ([`TabularConfig`](dpsynth/data_generation_v3.py)).
    *   **Relational (Multi-Table)**: Multi-table databases with foreign-key
        hierarchies and referential integrity
        ([`dpsynth.relational`](dpsynth/relational/synthesizer.py)).
    *   **Nested**: Hierarchically structured and typed records with shared and
        per-type schemas
        ([`dpsynth.experimental.nested`](dpsynth/experimental/nested.py)).
    *   **Sequence**: Event sequences, trajectories, and user session logs.
    *   **Text**: Unstructured text and prompt/response pairs via DP language
        model fine-tuning and conditional generation
        ([`dpsynth.text`](dpsynth/text/dp_sft.py)).
*   **Maturity and API Stability**: While the library supports these different
    modalities, they are currently at different levels of maturity. Basic
    tabular and discrete synthesis are the most mature and well-tested, whereas
    other modalities in experimental and research modules are still actively
    being developed, stress-tested, and improved. We expect the overall
    public-facing APIs (`MechanismConfig` and `CalibratedMechanism`) to remain
    mostly unchanged, and future changes should mostly (though not guaranteed)
    impact internal implementations rather than the public-facing APIs.
*   **Why Co-Development Matters**: State-of-the-art mechanisms for text data
    (such as [ACTG](https://arxiv.org/abs/2510.18232) and
    [MAPLE](https://arxiv.org/abs/2603.19258)) depend directly on
    state-of-the-art primitives for tabular data (for example, privately
    synthesizing structured metadata or summary attributes first and generating
    text conditioned on those attributes). Similarly, novel mechanisms for new
    modalities (including [nested](dpsynth/experimental/nested.py),
    [relational](https://arxiv.org/abs/2503.22970), sequence, and text data)
    can often be defined by reducing the problem to a tabular synthetic data
    problem. Having a robust library of tabular primitives makes it easy for
    other modalities to leverage them directly.
*   **Built for Heterogeneous, Mixed-Modality Data**: Many real-world datasets
    are *mixed* (for example, combining structured tabular columns with
    free-form text). DPSynth has been designed *from the ground up* to support
    such heterogeneous use cases, from how the
    [`MechanismConfig` / `CalibratedMechanism`](docs/mechanism_api.md)
    API is curated to how tight privacy accounting and budget calibration are
    performed under the hood across composed sub-mechanisms.
*   **Library Code vs. Binaries and Infrastructure**: As a library, DPSynth
    consists of importable Python code and does not house deployment binaries,
    experiment orchestration, or surrounding serving infrastructure in the core
    package. Instead, we maintain a dedicated [`examples/`](examples/) directory
    with runnable scripts and binaries that demonstrate how to use the library.

### Mechanism Architecture and Requirements

To keep the library cohesive across diverse modalities without constraining how
individual algorithms are built, mechanisms in DPSynth follow three core
principles:

1.  **Implement the `MechanismConfig` / `CalibratedMechanism` API**: Every
    mechanism is structured around two lightweight abstractions (detailed in
    [`docs/mechanism_api.md`](docs/mechanism_api.md)):
    *   `MechanismConfig`: A serializable, frozen configuration recipe that
        defines `configure(domain, zcdp_rho=...)` and
        `calibrate(domain, epsilon=..., delta=...)` to bind a domain and privacy
        budget and return a `CalibratedMechanism`.
    *   `CalibratedMechanism`: An immutable, runnable instance exposing
        `dp_event` (an exact characterization of privacy properties via
        `dp_accounting`) and `__call__(rng, data)` to execute on sensitive data.
    *   Because this contract places no restrictions on the types of `domain` or
        `data`, it is lightweight, modality-agnostic, and a natural fit for any
        DP mechanism: it cleanly separates configuration from execution and
        enables automatic privacy calibration and composition across mechanisms.
2.  **Provide a Carefully Curated Public API**: Mechanisms should expose a
    deliberate, minimal public surface (typically the config class, calibrated
    mechanism class, and any required domain/schema types) with a clear API
    contract, rather than exporting internal helpers or intermediate utilities
    to the top-level library.
3.  **Complete Flexibility of Internal Implementation**: Because end users and
    downstream pipelines depend only on the curated outer API contract, a
    mechanism's internal implementation details, modeling abstractions, and file
    organization are completely flexible and can evolve freely over time without
    breaking callers. If you have a novel synthetic data mechanism for any
    modality, consider contributing it upstream by
    [opening an issue](https://github.com/google/dpsynth/issues) to discuss the
    design (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Two Code Paths

DPSynth contains **two independent implementations** of differentially private
synthetic data generation. While both produce synthetic data using the same
underlying mathematical principles (marginal measurement + Private-PGM
inference), they were developed independently and have different trade-offs:

### 1. In-Memory (Local) Mode

**Entry point:** [`dpsynth.TabularConfig`](dpsynth/__init__.py) (backed by
[`data_generation_v3.py`](dpsynth/data_generation_v3.py))

Designed for **datasets that fit in memory** (e.g., Pandas DataFrames). We have
tested this on datasets up to ~100M rows, though performance will depend on the
number of attributes and domain sizes. This code path:

*   Operates directly on NumPy / Jax arrays and Pandas DataFrames via
    [`discrete_mechanisms/`](dpsynth/discrete_mechanisms/README.md).
*   Accepts domains and data as Python objects — no
    `DatasetDescriptor` required.
*   May be more feature-rich, including experimental mechanisms
    not yet ported to the pipeline mode.
*   Has limited scalability compared to the pipeline mode.

**CLI binary:** [`bin/main.py`](dpsynth/bin/README.md)

### 2. Scalable Pipeline Mode

**Entry point:**
[`data_generation.generate()`](dpsynth/data_generation.py)

Built for **large-scale data** that may not fit on a single machine. This code
path:

*   Runs on distributed frameworks (Apache Beam) via
    [`pipeline_dp.PipelineBackend`](https://github.com/OpenMined/pipeline-dp).
*   Uses
    [`pipeline_transformations/`](dpsynth/pipeline_transformations/README.md)
    for all DP operations.
*   Requires a
    [`DatasetDescriptor`](dpsynth/dataset_descriptors/README.md) to bridge
    format-specific data (CSV, TFRecord) with internal representations.
*   Also works in local settings (`pipeline_dp.LocalBackend`), but follows a
    different code path than the in-memory mode above.

**CLI binary:**
[`bin/run_data_generation.py`](dpsynth/bin/README.md)

### 3. Post-Processing Mode

**Entry point:**
[`postprocessing.generate_synthetic_data_from_marginals()`
](dpsynth/postprocessing.py)

For situations where **noisy marginals are already computed** by an
external system (e.g., a SQL pipeline or a custom DP mechanism), you
can bypass the measurement step entirely and use DPSynth purely for
post-processing. This code path takes pre-computed noisy marginals as
Pandas DataFrames. It automatically infers the domain from the
marginals, but requires categorical data.

### Differences Between the Two Main Code Paths

We have made efforts to align the APIs and behavior between the in-memory and
pipeline code paths, but because they were developed independently, there are
some differences:

*   **API surface:** The in-memory API accepts domains as a plain `dict[str,
    AttributeType]`, while the pipeline API uses the `DatasetDescriptor`
    abstraction.
*   **Budget accounting:** There may be small differences in how
    the privacy budget is split across sub-operations (derivation,
    compression, measurement).
*   **Feature availability:** The in-memory mode may support more experimental
    mechanisms or features that have not yet been ported to the pipeline mode.

> [!NOTE]
> If you observe significant differences in behavior or utility between the two
> code paths on the same dataset and parameters, please open an issue.

## Project Structure

### Shared Modules (Both Code Paths)

These modules are used by both the in-memory and pipeline code paths:

*   **[`domain.py`](dpsynth/domain.py)**: Public API for defining attribute
    domains (`CategoricalAttribute`, `NumericalAttribute`,
    `OpenSetCategoricalAttribute`). Users construct these objects to describe
    their data schema.
*   **[`constraints.py`](dpsynth/constraints.py)**: Definition and validation of
    cross-attribute constraints, provided by users to enforce structural
    properties.
*   **[`transformations.py`](dpsynth/transformations.py)**: Internal logic for
    encoding, discretization, and mapping values between domains.

### In-Memory Mode Only

*   **[`discrete_mechanisms/`](dpsynth/discrete_mechanisms/README.md)**: Local,
    single-machine DP mechanisms (AIM, MST, etc.) and shared mathematical
    utilities like domain compression.
*   **[`data_generation_v3.py`](dpsynth/data_generation_v3.py)**: The
    end-to-end in-memory generation pipeline. This is what
    `dpsynth.TabularConfig` exposes.
*   **[`local_mode/`](dpsynth/local_mode/)**: Locally-optimized DP primitives
    for quantiles and partition selection (NumPy/SciPy-based).
*   **[`adapters/pydantic.py`](dpsynth/adapters/pydantic.py)**: API for synthesizing
    collections of Pydantic models directly.

### Pipeline Mode Only

*   **[`dataset_descriptors/`](dpsynth/dataset_descriptors/README.md)**: The
    central orchestration layer. Bridges format-specific data (CSV, TFRecord)
    with internal mathematical representations.
*   **[`pipeline_transformations/`
    ](dpsynth/pipeline_transformations/README.md)**:
    Distributed Beam implementations of DP primitives, derivations, and final
    sample synthesis.
*   **[`data_generation.py`](dpsynth/data_generation.py)**: High-level API for
    generating synthetic data in data pipelines using `pipeline_dp`.
*   **[`diagnostic_info.proto`](dpsynth/diagnostic_info.proto)**: Proto
    definition for tracking DP accounting and utility metrics during pipeline
    execution.

### Post-Processing

*   **[`postprocessing.py`](dpsynth/postprocessing.py)**: Utilities for
    post-processing pre-computed noisy marginals into synthetic data via
    Private-PGM, without running DPSynth's own DP measurement step.

### Binaries & Tools

*   **[`bin/`](dpsynth/bin/README.md)**: Entry points for local prototyping
    (`main.py`) and distributed production jobs (`run_data_generation.py`).
*   **[`eval/`](dpsynth/eval/)**: Tabular evaluation engine for comparing real
    and synthetic data distributions.

## Which Code Path Should I Use?

| Scenario | Recommended |
|---|---|
| Fits in memory, Pandas workflow | **In-Memory** (`dpsynth.TabularConfig`) |
| Discrete data, precomputed marginals | **In-Memory** (`discrete_mechanisms`) |
| Large-scale, distributed processing | **Pipeline** (`data_generation`) |
| Marginals from an external system | **Post-Processing** |
| Prototyping / experimental features | **In-Memory** (more flexible) |

## Supported Synthesis Algorithms

Both code paths support the following DP mechanisms:

*   **AIM (Adaptive Iterative Mechanism)**: An MWEM-style algorithm that
    iteratively selects and measures low-dimensional marginals
    ([arXiv:2201.12677](https://arxiv.org/abs/2201.12677)).
*   **AIM-GDP**: A variant of AIM using Gaussian Differential Privacy accounting
    for tighter budget composition.
*   **MST (Maximum Spanning Tree)**: Computes an approximate maximum spanning
    tree over pairwise attribute correlations using the exponential mechanism
    ([arXiv:2108.04978](https://arxiv.org/abs/2108.04978)).
*   **SWIFT (Scalable Workload-Informed Factor Tree)**: An unpublished mechanism
    that operates on discrete data and improves over AIM for
    higher-dimensional datasets by supporting denser sets of marginal
    measurements. Numerical attributes can be handled via the existing
    discretization wrappers.
*   **INDEPENDENT**: A baseline mechanism that measures 1-way marginals and
    models all attributes independently.

## Further Documentation

Detailed guides are available in the [`docs/`](docs/)
directory:

*   **In-Memory DataFrame API Guide** (`docs/in_memory_api.md`):
    Detailed guide to using the Pandas-based API and local CLI.
*   **Scalable Pipeline API Guide** (`docs/scalable_beam_api.md`):
    Guide for distributed data generation.
*   **Mechanism API Architecture** (`docs/mechanism_api.md`):
    Two-stage configure-vs-calibrate pattern and composable mechanism contract.
*   **Privacy Accounting Philosophy & Architecture**
    (`docs/accounting_philosophy.md`): Neighboring relations, group
    privacy, adaptivity, privacy profiles (`PrivacyReport`), ill-formed input
    handling, and floating-point caveats.
*   **Data Model & Terminology** (`docs/data_and_terminology.md`):
    Attributes, schema specifications, and `domain.yaml` format.
*   **Processing Lifecycle** (`docs/processing_lifecycle.md`):
    The 5-stage mathematical lifecycle shared by both code paths.
*   **Contributor Guide** (`docs/contributors_guide.md`):
    Architecture, PipelineBackend programming rules, and evaluation framework.

*This is not an officially supported Google product. This project is
not eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).*
