<!-- Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# DPSynth

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

DPSynth is a Python library for generating differentially private (DP) synthetic
data. Given a sensitive dataset of records, DPSynth produces a synthetic dataset
that preserves the statistical properties, multi-dimensional correlations, and
structural relationships of the source data while satisfying formal
$(\epsilon, \delta)$-differential privacy.

## Supported Modalities and Scope

DPSynth is designed to support synthetic data generation across a broad range of
data modalities:

*   **Discrete**: Integer-encoded tabular datasets via marginal-based graphical
    model mechanisms (such as AIM, MST, SWIFT, and Independent).
*   **Tabular**: Single-table schemas with mixed categorical, numerical, and
    open-set string attributes.
*   **Relational**: Multi-table databases with foreign-key hierarchies and
    referential integrity.
*   **Nested**: Hierarchically structured and typed records with shared and
    per-type schemas.
*   **Sequence**: Event sequences, trajectories, and user session logs.
*   **Text**: Unstructured text and prompt-response pairs via DP language model
    fine-tuning and attribute-conditioned generation.

Co-developing these modalities in a single library is a deliberate design
choice. State-of-the-art mechanisms for unstructured text (such as
[ACTG](https://arxiv.org/abs/2510.18232) and
[MAPLE](https://arxiv.org/abs/2603.19258)) rely on tabular synthesis primitives
to privately model structured metadata or summary attributes first, then
generate text conditioned on those synthetic attributes. Similarly, mechanisms
for [relational](https://arxiv.org/abs/2503.22970), nested, and sequence data
frequently decompose complex structures into a series of tabular synthesis
subproblems, and many real-world datasets combine structured columns with
free-form text in a single schema.

To make mechanisms composable across modalities, every algorithm in DPSynth
follows the same three-step workflow (detailed in
[Mechanism API](mechanism_api.md) and
[Privacy Accounting Philosophy](accounting_philosophy.md)):

1.  **Configure**: Instantiate a `MechanismConfig` specifying algorithm
    hyperparameters.
2.  **Calibrate**: Call
    `dpsynth.calibrate(config, domain, epsilon=..., delta=...)` to bind the data
    domain and target privacy budget, returning an immutable
    `CalibratedMechanism` with an exact `dp_accounting` guarantee.
3.  **Run**: Call `mechanism(rng, data)` on the sensitive dataset to generate
    synthetic records.

While all modalities share this core contract, they are currently at different
stages of maturity: discrete and single-table tabular synthesis are the most
mature and well-tested, whereas newer modalities in experimental and research
modules are actively being developed and expanded.

## Quickstart

For tabular datasets that fit in single-machine memory, `dpsynth.TabularConfig`
operates directly on Pandas DataFrames. See
[Specifying Attribute Domains](specifying_domains.md) for how to construct or
infer attribute domains for your schema. In the example below, we load the
predefined domain specification for the UCI Adult dataset from
[adult_domain.yaml](https://github.com/google/dpsynth/blob/main/examples/adult_domain.yaml):

```python
import dpsynth
from dpsynth import discrete_mechanisms
import numpy as np
import pandas as pd

domains = dpsynth.from_yaml("examples/adult_domain.yaml")
sensitive_df = pd.read_csv("adult.csv")

config = dpsynth.TabularConfig(
    discrete_mechanism=discrete_mechanisms.MSTConfig(),
)
mechanism = dpsynth.calibrate(config, domains, epsilon=1.0, delta=1e-6)
result = mechanism(np.random.default_rng(), sensitive_df)
synthetic_df = result.synthetic_data
```

For large-scale tabular datasets that require distributed execution across a
cluster via
Apache Beam,
see the
[Scalable Beam API Guide](scalable_beam_api.md).

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

Overview <self>
api_reference
```

```{toctree}
:maxdepth: 2
:caption: Tabular Data
:hidden:

specifying_domains
processing_lifecycle
scalable_pipeline_api
```

```{toctree}
:maxdepth: 2
:caption: Developer Documentation
:hidden:

mechanism_api
accounting_philosophy
contributors_guide
```
