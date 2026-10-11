# Sitemap

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="index.md">Overview & Home</a></summary>

*   [Why DPSynth?](index.md#why-dpsynth)
*   [Core APIs and Execution Models](index.md#core-apis-and-execution-models)
    *   [1. In-Memory DataFrame API (`dpsynth.TabularConfig`)](index.md#1-in-memory-dataframe-api-dpsynth-tabularconfig)
    *   [2. Scalable PipelineBackend API (`dpsynth.data_generation`)](index.md#2-scalable-pipelinebackend-api-dpsynthdata_generation)
*   [Documentation Sitemap & Navigation](index.md#documentation-sitemap--navigation)
*   [Supported Synthesis Algorithms](index.md#supported-synthesis-algorithms)
*   [Quick Start Example (CLI)](index.md#quick-start-example-cli)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="specifying_domains.md">Specifying Attribute Domains</a></summary>

*   [Why Domains Must Be Specified Up Front](specifying_domains.md#why-domains-must-be-specified-up-front)
*   [Supported Attribute Types](specifying_domains.md#supported-attribute-types)
    *   [Out-of-Domain Handling](specifying_domains.md#out-of-domain-handling)
*   [Specifying Domains](specifying_domains.md#specifying-domains)
    *   [1. Directly in the DPSynth Language](specifying_domains.md#1-directly-in-the-dpsynth-language)
    *   [2. From Pydantic Models](specifying_domains.md#2-from-pydantic-models)
    *   [3. From Protocol Buffers](specifying_domains.md#3-from-protocol-buffers)
    *   [4. From Pandas Dtypes](specifying_domains.md#4-from-pandas-dtypes)
    *   [Customizing Inferred Domains](specifying_domains.md#customizing-inferred-domains)
*   [Tabular Schemas vs. Discrete Domains](specifying_domains.md#tabular-schemas-vs-discrete-domains)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="data_and_terminology.md">Data Model and Terminology</a></summary>

*   [Core Terminology](data_and_terminology.md#core-terminology)
*   [Supported Data Specifications](data_and_terminology.md#supported-data-specifications)
    *   [1. Flat Tabular Structure](data_and_terminology.md#1-flat-tabular-structure)
    *   [2. Column Data Types](data_and_terminology.md#2-column-data-types)
    *   [3. Record Independence (Differential Privacy Assumption)](data_and_terminology.md#3-record-independence-differential-privacy-assumption)
*   [Supported Attribute Classifications](data_and_terminology.md#supported-attribute-classifications)
    *   [1. `CategoricalAttribute` (Known Finite Domain)](data_and_terminology.md#1-categoricalattribute-known-finite-domain)
    *   [2. `OpenSetCategoricalAttribute` (Unknown Categorical Domain)](data_and_terminology.md#2-opensetcategoricalattribute-unknown-categorical-domain)
    *   [3. `NumericalAttribute` (Continuous or Integer Range)](data_and_terminology.md#3-numericalattribute-continuous-or-integer-range)
    *   [4. `FreeFormTextAttribute` (Unstructured Text)](data_and_terminology.md#4-freeformtextattribute-unstructured-text)
*   [Writing a `domain.yaml` Specification](data_and_terminology.md#writing-a-domainyaml-specification)
    *   [YAML Syntax & Parsing Rules](data_and_terminology.md#yaml-syntax--parsing-rules)
    *   [Example: `adult_domain.yaml`](data_and_terminology.md#example-adult_domainyaml)
*   [Creating Domains Natively in Python](data_and_terminology.md#creating-domains-natively-in-python)
*   [Automatic Schema Deduction & Population](data_and_terminology.md#automatic-schema-deduction--population)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="in_memory_api.md">In-Memory DataFrame API Guide</a></summary>

*   [Python API: `dpsynth.TabularConfig`](in_memory_api.md#python-api-dpsynth-tabularconfig)
    *   [Function Signature](in_memory_api.md#function-signature)
    *   [Key Arguments](in_memory_api.md#key-arguments)
*   [End-to-End Python Example](in_memory_api.md#end-to-end-python-example)
*   [Command-Line Interface: `bin/main.py`](in_memory_api.md#command-line-interface-binmainpy)
    *   [CLI Execution Syntax](in_memory_api.md#cli-execution-syntax)
    *   [Supported CLI Flags](in_memory_api.md#supported-cli-flags)
*   [Under the Hood: The In-Memory Lifecycle](in_memory_api.md#under-the-hood-the-in-memory-lifecycle)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="scalable_beam_api.md">Scalable Beam API Guide</a></summary>

*   [Python API: `data_generation.generate`](scalable_beam_api.md#python-api-data_generationgenerate)
*   [Creating a `DatasetDescriptor`](scalable_beam_api.md#creating-a-datasetdescriptor)
    *   [Configuration: `DataGenerationConfig`](scalable_beam_api.md#configuration-datagenerationconfig)
    *   [Running the Pipeline in Python](scalable_beam_api.md#running-the-pipeline-in-python)
*   [Supported I/O Connectors](scalable_beam_api.md#supported-io-connectors)
*   [Command-Line Interface: `bin/run_data_generation.py`](scalable_beam_api.md#command-line-interface-binrun_data_generationpy)
    *   [CLI Execution Syntax (Distributed Cluster)](scalable_beam_api.md#cli-execution-syntax-distributed-cluster)
    *   [Comprehensive Flag Reference](scalable_beam_api.md#comprehensive-flag-reference)
*   [Specifying Attribute Domains (`--domain_file`)](scalable_beam_api.md#specifying-attribute-domains-domain_file)
*   [Saving & Re-Using the Trained Model](scalable_beam_api.md#saving--re-using-the-trained-model)
    *   [1. How to Save the Model](scalable_beam_api.md#1-how-to-save-the-model)
    *   [2. How to Load and Sample from a Saved Model](scalable_beam_api.md#2-how-to-load-and-sample-from-a-saved-model)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="processing_lifecycle.md">Processing and Synthesis Lifecycle</a></summary>

*   [Mathematical Modeling & Discretization](processing_lifecycle.md#mathematical-modeling--discretization)
*   [Stage 1: Derivation and Initialization](processing_lifecycle.md#stage-1-derivation-and-initialization)
    *   [Continuous Numerical Discretization](processing_lifecycle.md#continuous-numerical-discretization)
    *   [Open-Set Categorical Discovery](processing_lifecycle.md#open-set-categorical-discovery)
*   [Stage 2: Integer Encoding](processing_lifecycle.md#stage-2-integer-encoding)
*   [Stage 3: Domain Compression](processing_lifecycle.md#stage-3-domain-compression)
*   [Stage 4: Graphical Model Inference (Private-PGM)](processing_lifecycle.md#stage-4-graphical-model-inference-private-pgm)
    *   [1. Workload Measurement](processing_lifecycle.md#1-workload-measurement)
    *   [2. Private-PGM Inference](processing_lifecycle.md#2-private-pgm-inference)
*   [Stage 5: Sampling, Uncompression, & Decoding](processing_lifecycle.md#stage-5-sampling-uncompression--decoding)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="contributors_guide.md">Contributor and Architecture Guide</a></summary>

*   [Codebase Architecture & Directory Structure](contributors_guide.md#codebase-architecture--directory-structure)
    *   [Architectural Separation: General vs. Distributed](contributors_guide.md#architectural-separation-general-vs-distributed)
*   [Core Orchestration Abstractions](contributors_guide.md#core-orchestration-abstractions)
*   [Guide for Writing Pipelines with `PipelineBackend`](contributors_guide.md#guide-for-writing-pipelines-with-pipelinebackend)
    *   [1. Why Framework-Agnostic?](contributors_guide.md#1-why-framework-agnostic)
    *   [2. Designing Pipeline Functions](contributors_guide.md#2-designing-pipeline-functions)
    *   [3. Core `PipelineBackend` Methods Available](contributors_guide.md#3-core-pipelinebackend-methods-available)
    *   [4. Critical Constraints & Safety Guardrails](contributors_guide.md#4-critical-constraints--safety-guardrails)
    *   [5. PipelineDP Integration & DP Aggregations](contributors_guide.md#5-pipelinedp-integration--dp-aggregations)
*   [Diagnostic Information & DP Accounting](contributors_guide.md#diagnostic-information--dp-accounting)
*   [Marginal-Based Inference (`mbi`)](contributors_guide.md#marginal-based-inference-mbi)
    *   [1. Domain & Data Abstractions](contributors_guide.md#1-domain--data-abstractions)
    *   [2. Markov Random Field (`mbi.MarkovRandomField`)](contributors_guide.md#2-markov-random-field-mbimarkovrandomfield)
    *   [3. Mirror Descent Optimization & Private-PGM Solver](contributors_guide.md#3-mirror-descent-optimization--private-pgm-solver)
*   [The Tabular Evaluation Framework (`eval/`)](contributors_guide.md#the-tabular-evaluation-framework-eval)
    *   [Evaluation Metrics & Workflow](contributors_guide.md#evaluation-metrics--workflow)
    *   [Running Evaluation (CLI)](contributors_guide.md#running-evaluation-cli)

</details>

--------------------------------------------------------------------------------

<details>
<summary>📁 <a href="mechanism_api.md">Mechanism API Architecture</a></summary>

*   [Design Decisions: `MechanismConfig` vs. `CalibratedMechanism`](mechanism_api.md#design-decisions)
*   [The 3-Step Mechanism Pipeline](mechanism_api.md#three-step-pipeline)
*   [`configure(zcdp_rho)` vs. `calibrate(epsilon, delta)`](mechanism_api.md#configure-vs-calibrate)
*   [Notes on the `configure(zcdp_rho)` Design](mechanism_api.md#zcdp-intermediate)
*   [`zcdp_rho` vs. Natural Privacy Parameters](mechanism_api.md#natural-parameters)

</details>
