# Data Model and Terminology

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

To generate high-fidelity synthetic data while satisfying differential privacy,
**DPSynth** relies on precise metadata definitions for every attribute in your
dataset. This page introduces core terminology for structured schemas, explains
the attribute types supported by `dpsynth.domain`, and details how to structure
a `domain.yaml` specification.

--------------------------------------------------------------------------------

## Core Terminology

*   **Record**: A single entity or tuple of values in a dataset (e.g., a row in
    a CSV file or SQL table, or an individual Protocol Buffer message).
*   **Attribute (Column)**: A distinct feature across records (e.g., `age`,
    `education`, `income`).
*   **Domain**: The mathematical support or set of valid values for an
    attribute. Defining the domain bounds privacy sensitivity and limits the
    state space for graphical model inference.
*   **Discretization**: The process of mapping continuous numerical ranges or
    unbounded categorical string spaces into finite, discrete integer bins `[0,
    K-1]`.

--------------------------------------------------------------------------------

## Input Data Format (Single-Table Tabular APIs)

While DPSynth also provides dedicated mechanisms for multi-table relational
data (`dpsynth.relational`) and hierarchical nested data
(`dpsynth.experimental.nested`), the single-table tabular APIs
(`dpsynth.TabularConfig` and `dpsynth.data_generation`) work with structured
data satisfying the following requirements:

### 1. Flat Tabular Structure

*   **Single-Table**: For the single-table tabular APIs, the dataset must
    represent a single flat table of records (for multi-table star or snowflake
    schemas, use `dpsynth.relational.MultiTableConfig`).
*   **No Hierarchical Arrays**: In the distributed tabular pipeline, columns
    containing **repeated fields, arrays, or lists** (e.g., a repeated proto
    field or JSON lists) are ignored during parsing (for 1-to-many nested lists,
    see `dpsynth.experimental.nested`).
*   **No Unions**: Protobuf `oneof` fields or dynamically typed union columns
    are **unsupported**.

### 2. Column Data Types

Each column must map to one of the following supported scalar data types: *
**Integer (`INT`)**: Ordinal or discrete integer keys. * **Float (`FLOAT`)**:
Continuous floating-point values. * **String (`STR`)**: Closed-set or open-set
string categories, or free-form text. * **Boolean (`BOOL`)**: True/False binary
flags. * **Enum (`ENUM`)**: Named integer-backed categorical categories
(Protobuf enums).

### 3. Record Independence (Differential Privacy Assumption)

It is assumed that each **record** comes from different **privacy unit**.

## Supported Attribute Classifications

DPSynth classifies attributes in `dpsynth.domain` into four types:

### 1. `CategoricalAttribute` (Known Finite Domain)

Used when the complete set of valid categories is known in advance (e.g., days
of the week, US states, or boolean flags).

### 2. `OpenSetCategoricalAttribute` (Unknown Categorical Domain)

Used for categorical string columns where the exact list of unique categories is
unbounded or unknown prior to inspection (e.g., user job titles or city names).
DPSynth can deduce values with Differential Privacy.

### 3. `NumericalAttribute` (Continuous or Integer Range)

Used for continuous floating-point values or ordered integers (e.g., age,
salary, transaction amounts).

### 4. `FreeFormTextAttribute` (Unstructured Text)

Used for unstructured natural language fields (e.g., reviews, messages, or
notes) that can be synthesized alone or jointly alongside structured tabular
attributes.

--------------------------------------------------------------------------------

## Writing a `domain.yaml` Specification

`domain.yaml` contains specification of Data Scheme. The specification of the
YAML file is required for in-memory CLI interface.
For Beam API, the YAML file is required
only when numerical attributes are used.

### YAML Syntax & Parsing Rules

*   If `possible_values` is present $\rightarrow$ parsed as
    `CategoricalAttribute`.
*   If `min_value` and `max_value` are present $\rightarrow$ parsed as
    `NumericalAttribute`.
*   If the mapping is empty (`{}`) or `None` $\rightarrow$ parsed as
    `OpenSetCategoricalAttribute`.

### Example: `adult_domain.yaml`

```yaml
# Categorical column with explicitly known categories
workclass:
  possible_values:
    - "?" # Index 0: Fallback for missing values
    - "Private"
    - "Self-emp-not-inc"
    - "Self-emp-inc"
    - "Federal-gov"
    - "Local-gov"
    - "State-gov"
  out_of_domain_index: 0

# Numerical integer column with known upper and lower bounds
age:
  min_value: 17.0
  max_value: 90.0
  dtype: "int"
  clip_to_range: true

# Numerical continuous floating-point column
capital_gain:
  min_value: 0.0
  max_value: 99999.0
  dtype: "float"
  clip_to_range: true

# Open-set categorical column with unknown unique categories
occupation: {}

# Boolean column
income_bracket_gt_50k:
  possible_values:
    - false
    - true
```

### Creating Domains Natively in Python

Instead of loading a YAML file, you can also construct your schema domain
dictionary directly in Python. This is particularly useful when programmatically
generating constraints or loading categories dynamically from metadata stores.

```python
from dpsynth import domain

attribute_domains = {
    # 1. CategoricalAttribute: known finite set of unique categories
    "workclass": domain.CategoricalAttribute(
        possible_values=["?", "Private", "Self-emp-not-inc", "Federal-gov"],
        out_of_domain_index=0  # Maps out-of-domain entries to index 0 ("?")
    ),

    # 2. NumericalAttribute: continuous floating-point or integer bounds
    "age": domain.NumericalAttribute(
        min_value=17.0,
        max_value=90.0,
        dtype="int",
        clip_to_range=True
    ),

    # 3. OpenSetCategoricalAttribute: dynamic private categorical discovery
    "occupation": domain.OpenSetCategoricalAttribute(
        default_value="Unknown"
    )
}
```

--------------------------------------------------------------------------------

## Automatic Schema Deduction & Population

If you are using the [Scalable PipelineBackend API](scalable_beam_api.md) with structured formats like TFRecord, you do not always need to specify an exhaustive YAML file upfront.

1.  **Format Deduction**: DPSynth inspects a sample of records
    to deduce column names and native types (`INT`, `STR`,
    `FLOAT`).
2.  **The Population Phase**: The pipeline runs distributed DP transformations
    ([`pipeline_transformations/`](../pipeline_transformations/README.md))
    to privately compute quantiles for numerical columns and discover valid
    categories for strings, fully populating the internal `DatasetDescriptor`
    automatically.
