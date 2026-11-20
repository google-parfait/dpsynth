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

# Examples

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

DPSynth is designed to be highly flexible across data modalities, input
representations, and synthesis workflows. Rather than assuming a single data
format or modeling strategy, the
[examples directory](https://github.com/google-parfait/dpsynth/tree/main/examples)
contains focused, runnable scripts that each showcase a different capability of
the library, from in-memory DataFrame and distributed Beam synthesis to
precomputed marginal workflows, neural sequence modeling, language model
fine-tuning, and extracting structured insights from unstructured text.

To keep the repository lightweight, datasets are not checked into Git. Companion
scripts in
[examples/datasets](https://github.com/google-parfait/dpsynth/tree/main/examples/datasets)
download and prepare public reference datasets so every example runs out of the
box, and each script exposes a small dataclass-based command-line interface for
custom dataset paths and privacy budgets.

| Example | Modality | Capability Showcased | Default Dataset |
| :--- | :--- | :--- | :--- |
| [tabular_synthesis.py](https://github.com/google-parfait/dpsynth/blob/main/examples/tabular_synthesis.py) | Tabular | Mixed DataFrame synthesis | UCI Adult |
| [pipeline_tabular_synthesis.py](https://github.com/google-parfait/dpsynth/blob/main/examples/pipeline_tabular_synthesis.py) | Tabular | Distributed Beam pipeline synthesis | UCI Adult |
| [discrete_synthesis.py](https://github.com/google-parfait/dpsynth/blob/main/examples/discrete_synthesis.py) | Discrete | Integer arrays or precomputed marginals | 1940 US Census |
| [postprocess_marginals.py](https://github.com/google-parfait/dpsynth/blob/main/examples/postprocess_marginals.py) | Discrete / Tabular | Post-processing external DP marginals | 1940 US Census |
| [tabular_transformer.py](https://github.com/google-parfait/dpsynth/blob/main/examples/tabular_transformer.py) | Sequence / Tabular | Neural Transformer in TabularConfig | MSNBC Clickstream |
| [finetune_text.py](https://github.com/google-parfait/dpsynth/blob/main/examples/finetune_text.py) | Free-Form Text | DP LLM fine-tuning and auditing | PubMed Abstracts |
| [annotate_text.py](https://github.com/google-parfait/dpsynth/blob/main/examples/annotate_text.py) + [tabular_synthesis.py](https://github.com/google-parfait/dpsynth/blob/main/examples/tabular_synthesis.py) | Text to Tabular | Structured insights from unstructured text | WildChat |

## Tabular Synthesis

The tabular synthesis script showcases the standard in-memory entry point for
single-table CSV or Parquet datasets with mixed categorical and numerical
columns. Given a declarative YAML domain schema, DPSynth calibrates all stages
to a target privacy budget up front, performs differentially private quantile
binning and category encoding, runs a marginal-based mechanism (SWIFT), decodes
synthetic records back to the original column types, and writes both the
synthetic table and a YAML privacy report.

Download the 48,842-row UCI Adult dataset with
[download_adult.py](https://github.com/google-parfait/dpsynth/blob/main/examples/datasets/download_adult.py)
and run the script from the `examples` directory:

```bash
cd examples
python datasets/download_adult.py
python tabular_synthesis.py \
    --input_path=datasets/adult.csv \
    --domain_path=datasets/adult_domain.yaml \
    --output_dir=dpsynth_output \
    --epsilon=1.0 \
    --delta=1e-6
```

## Distributed Pipeline Synthesis

The pipeline tabular synthesis script (requiring `dpsynth[pipeline]`) showcases
scaling single-table synthesis to datasets that exceed single-machine memory via
the `dpsynth.pipeline` API and PipelineDP. Rather than materializing a Pandas
DataFrame, records stream through an Apache Beam pipeline to compute
differentially private marginals and write synthetic CSV output, with an option
to execute over a plain Python iterable (`--no-use_beam`) for lightweight local
debugging:

```bash
cd examples
python datasets/download_adult.py
python pipeline_tabular_synthesis.py \
    --input_path=datasets/adult.csv \
    --domain_path=datasets/adult_domain.yaml \
    --output_dir=dpsynth_output \
    --epsilon=1.0 \
    --delta=1e-6
```

## Discrete Synthesis

Whereas tabular data consists of DataFrames with raw mixed column types
(strings, continuous numbers, open-set vocabularies, and missing values) that
require differentially private quantile binning, category encoding, and decoding
back to the original schema, discrete data is already integer-encoded over known
finite attribute sizes $\{0, \dots, n_i - 1\}$ with no missing values. Discrete
mechanisms accept either an integer-coded record array (`mbi.Dataset`) or an
archive of precomputed exact marginal counts (`mbi.CliqueVector`), bypassing
DataFrame preprocessing and decoding altogether.

The discrete synthesis script showcases running a marginal-based mechanism
directly on precomputed exact marginals. Unlike pure post-processing, the
mechanism still privately selects which marginals to measure from the supporting
marginals in the archive. On the full-count 1940 US Census (131.9M rows across
135 columns), working from the 415 MB precomputed marginal archive downloaded by
[download_census_marginals.py](https://github.com/google-parfait/dpsynth/blob/main/examples/datasets/download_census_marginals.py)
instead of the 54 GB raw extract saves substantial pre- and post-processing time
and memory:

```bash
cd examples
python datasets/download_census_marginals.py
python discrete_synthesis.py \
    --input_path=datasets/census_1940_clique_vector.npz \
    --output_dir=dpsynth_output \
    --epsilon=1.0 \
    --delta=1e-6
```

## Marginal Post-Processing

The marginal post-processing script sits outside DPSynth's Mechanism API and
showcases generating consistent synthetic records purely from noisy DP marginal
tables that were already computed and privatized upstream (for example, by a
distributed SQL engine). Fitting a graphical model with MBI's Universal
Accelerated Method and sampling records consumes no additional privacy budget
and operates on 2.2 MB of Census marginal CSVs rather than the 54 GB raw
dataset. The script adds Gaussian noise inline only to keep the example
self-contained; in practice, the marginals loaded from disk would already be
privatized:

```bash
cd examples
python datasets/download_census_marginals.py
python postprocess_marginals.py \
    --marginals_dir=datasets/census_marginals \
    --output_dir=dpsynth_output \
    --sigma=50.0 \
    --num_rows=10000
```

## Tabular Transformer

The tabular transformer script (requiring `dpsynth[training]`) showcases how a
neural generative model trained with BandMF plugs into the same tabular
configuration workflow as a drop-in replacement for marginal-based mechanisms.
The outer tabular wrapper handles schema encoding, calibration, and decoding
while an autoregressive Transformer learns sequential dependencies across
columns, making it well suited for ordered event sequences such as the 16-step
MSNBC clickstream dataset prepared by
[download_msnbc.py](https://github.com/google-parfait/dpsynth/blob/main/examples/datasets/download_msnbc.py):

```bash
cd examples
python datasets/download_msnbc.py
python tabular_transformer.py \
    --input_path=datasets/msnbc.csv \
    --domain_path=datasets/msnbc_domain.yaml \
    --output_dir=dpsynth_output \
    --epsilon=1.0 \
    --delta=1e-6
```

## Language Model Fine-Tuning

The text fine-tuning script (requiring `dpsynth[training]`) showcases generating
synthetic unstructured text by fine-tuning Gemma LoRA weights on private
documents with BandMF. Automatic device-mesh sharding lets the same binary run
without modification on a single GPU or across a large cluster of GPUs or TPUs.
Each document is paired with a shared instruction prompt during training,
synthetic documents are sampled afterward as free post-processing alongside a
saved model checkpoint, and optional canary documents (generated with
[generate_pubmed_canaries.py](https://github.com/google-parfait/dpsynth/blob/main/examples/datasets/generate_pubmed_canaries.py))
produce an empirical memorization audit report alongside the formal privacy
guarantee:

```bash
cd examples
python datasets/download_pubmed.py
python finetune_text.py \
    --data_path=datasets/pubmed.json \
    --canary_path=datasets/pubmed_canaries.json \
    --output_dir=dpsynth_output \
    --instruction="Write the abstract of a research paper."
```

## Structured Insights from Unstructured Text

This example combines
[annotate_text.py](https://github.com/google-parfait/dpsynth/blob/main/examples/annotate_text.py)
and
[tabular_synthesis.py](https://github.com/google-parfait/dpsynth/blob/main/examples/tabular_synthesis.py)
to extract differentially private structured insights from raw unstructured
text. Defining the attributes of interest as a Pydantic model lets the Pydantic
adapter infer a companion DPSynth domain YAML automatically, while Gemini
annotates each WildChat conversation independently into a structured Parquet
feature table that tabular synthesis then privatizes end to end.

First, download the WildChat dataset with
[download_wildchat.py](https://github.com/google-parfait/dpsynth/blob/main/examples/datasets/download_wildchat.py)
and annotate the conversations into a structured feature table. Running
`annotate_text.py` requires a `google.genai` API key (`GEMINI_API_KEY`), and you
can skip that step if you want since the download script also fetches a
pre-annotated feature table:

```bash
cd examples
python datasets/download_wildchat.py
python annotate_text.py \
    --input_path=datasets/wildchat.parquet \
    --output_path=datasets/wildchat_features.parquet \
    --domain_path=datasets/wildchat_domain.yaml
```

Then run `tabular_synthesis.py` on the feature table and companion domain YAML
to generate differentially private synthetic records:

```bash
python tabular_synthesis.py \
    --input_path=datasets/wildchat_features.parquet \
    --domain_path=datasets/wildchat_domain.yaml
```

## Interactive Notebooks

If you prefer an interactive walkthrough in Jupyter or Colab, start with
[quickstart.ipynb](https://github.com/google-parfait/dpsynth/blob/main/examples/quickstart.ipynb)
for a concise tour of the DataFrame API, or explore
[detailed_example_theory_and_in_memory_api.ipynb](https://github.com/google-parfait/dpsynth/blob/main/examples/detailed_example_theory_and_in_memory_api.ipynb)
for a deeper dive into domain specification, structural constraints, and privacy
accounting.
