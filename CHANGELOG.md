# Changelog

All notable changes to the `dpsynth` library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

-   **API stabilization (`MechanismConfig` / `CalibratedMechanism`)**: Unified
    the mechanism lifecycle into a clean, predictable workflow: construct a
    configuration recipe (`MechanismConfig`), bind privacy parameters via
    `calibrate(schema, epsilon=..., delta=...)` to produce an immutable
    `CalibratedMechanism`, and run on data. Accompanied by the new
    `dpsynth.Schema` dataclass for attribute specs and cross-attribute
    constraints, accepted directly by `calibrate()` and `configure()` to enable
    reusable presets decoupled from individual datasets.
-   **Relational (multi-table) synthesis**: Exposed `MultiTableConfig`,
    `MultiTableMechanism`, and `ForeignKeyRelation` at top-level API (`dpsynth`
    and `dpsynth.relational`). Enables private synthesis of relational databases
    with foreign-key integrity (`ForeignKeyRelation`), including an end-to-end
    California Census housing example.
-   **Nested tabular synthesis**: Added `NestedTabularSynthesizer` for
    typed/nested tabular data where records share common attributes but have
    type-specific sub-schemas, using a two-model shared/per-type architecture.
-   **YAML serialization**: New `dpsynth.serialize` module with
    `dpsynth.to_yaml()` and `dpsynth.from_yaml()` for human-readable
    serialization of mechanism configs, calibration results, and
    `dp_accounting.DpEvent` objects.
-   **Composition-over-inheritance architecture**: Refactored discrete mechanism
    implementations to improve code-reuse while ensuring mechanism
    implementations can be verified in isolation (without understanding and
    traversing complex inheritance hierarchies).
-   **`calibrate()` enhancements (schema, Poisson sampling, user bounding)**:
    `MechanismConfig.calibrate()` now accepts an optional `schema` (or
    `domain`), `poisson_sampling_prob` (accounting for subsampling amplification
    via PLD/RDP), and `max_records_per_user` (scaling sensitivity for
    user-level differential privacy).
-   **Synthetic text generation example**: Added end-to-end `finetune_pubmed.py`
    example demonstrating DP fine-tuning for synthetic text generation using JAX
    Privacy under the hood.
-   **Tabular Transformer synthesizer (experimental)**: Added an experimental
    generative Tabular Transformer in Flax NNX
    (`dpsynth.experimental.synthetic_transformer`). It trains an autoregressive
    causal Transformer over tabular attributes using DP-SGD powered by JAX
    Privacy (`DPTrainer`), and implements DPSynth's `CalibratedMechanism`
    interface to plug directly into `TabularSynthesizer`.
-   **JAX acceleration & column compression**: Optional `use_jax_for_bincount`
    and `use_jax_for_generation` flags in `TabularConfig` / `DiscreteConfig`
    greatly improving efficiency over the default numpy implementations in very
    large scale settings.
-   **`compress_columns` in `TabularConfig`**: Automatically compresses rare
    categories (< 3σ) for categorical columns not present in constraints.

### Internal / Cleanups

-   **Decoupled initializers from MBI**: Removed MBI and JAX dependencies from
    `adapters/beam.py` worker execution paths.
-   **Dependency management with `uv` and `pylock.toml`**: Added PEP 751
    lockfile and `uv` setup for contributors, enforced in CI.

## [0.4.0] - 2026-08-26

### Removed

-   **Breaking**: Removed deprecated `zcdp_rho` argument from
    `MechanismConfig.calibrate()`. Use `MechanismConfig.configure(zcdp_rho=...)`
    directly instead.
-   **Breaking**: Removed deprecated `dpsynth.data_generation_v2` module.
    Use `dpsynth.TabularConfig` instead.

### Changed

-   **Breaking**: The base install now contains only the dependencies required
    for in-memory synthesis via `dpsynth.TabularSynthesizer`. Heavyweight and
    feature-specific dependencies moved to optional extras to keep the base
    install small:
    -   `[pipeline]` — scalable Apache Beam / `pipeline_dp` execution path and
        its TensorFlow-based TFRecord I/O.
    -   `[text]` — free-form text features (`dpsynth.text`): GenAI batch feature
        extraction and DP fine-tuning of Gemma. `google-genai` moved here from
        the base install.
    -   `[examples]` — dependencies used only by the example notebooks
        (`kagglehub`, `scikit-learn`, `sdmetrics`).
    -   `[all]` — the `pipeline` and `text` feature extras.

    DPSynth is installed from GitHub rather than PyPI, so request extras with
    the direct-reference syntax, e.g.:

    ```
    pip install "dpsynth[pipeline] @ git+https://github.com/google-parfait/dpsynth.git"
    ```

    Users of the Beam pipeline should include the `pipeline` extra.

## [0.1.0] - 2026-06-15

Initial public release of DP Synth — a library for generating differentially
private synthetic data.

### Added

This first release contains code for generating differentially private synthetic
tabular data using marginal measurement and Private-PGM inference, including:

-   **Two execution modes**: In-memory local mode
    (via `dpsynth.TabularSynthesizer`, tested up to ~100M rows) and a
    workloads.
-   **Marginal-based mechanisms**: AIM, MST, Independent, and Direct mechanisms
    for selecting and measuring marginals under differential privacy.
-   **Closed-domain categorical attributes**: Standard categorical columns
    where the full domain is known upfront.
-   **Open-domain categorical attributes**: DP partition selection to privately
    discover significant categories when the domain is not known in advance.
-   **Numerical attributes**: Discretization with configurable
    `interval_handling` to control how intervals are converted back to values
    (`midpoint`, `sample`, or raw `pd.Interval`).
-   **Quickstart notebook**: Interactive Colab notebook demonstrating basic
    usage of the library.
-   **Documentation**: README with architecture overview, module-level READMEs,
    and work-in-progress notice.

[0.1.0]: https://github.com/google-parfait/dpsynth/releases/tag/v0.1.0
