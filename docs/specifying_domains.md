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

# Specifying Attribute Domains

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

Every synthesis mechanism in DPSynth requires an explicit domain specification
describing the attributes (columns) of your dataset and the set of valid values
each attribute can take. This guide explains why domains must be defined before
touching private data, how to express domains directly in the DPSynth domain
language, and how to derive them automatically from structured schema
definitions like Pydantic models, Protocol Buffers, and Pandas dtypes.

## Why Domains Must Be Specified Up Front

In differential privacy, the domain of an attribute defines the sample space of
noisy measurements and bounds the sensitivity of each record's contribution.
Because computing minimum and maximum values or extracting unique categories
directly from private data is not differentially private, the domain must be
specified before inspecting the sensitive dataset. Adding or removing a single
outlier record can change an observed numerical range or reveal the presence of
a rare category.

In DPSynth's [three-step mechanism
lifecycle](mechanism_api.md#three-step-pipeline), you pass the domain to the
calibration step to allocate privacy budgets and construct per-column
initializers before the calibrated mechanism ever sees private data. When the
valid categories of a string column are not known from public metadata, you can
use an open-set categorical attribute rather than scanning the raw data; DPSynth
will automatically spend a fraction of the privacy budget on differentially
private partition selection to discover common categories.

## Supported Attribute Types

DPSynth provides four attribute classes in the domain module to model tabular
columns:

| Attribute Type | When to Use | Required Public Metadata | DP Initialization |
| :--- | :--- | :--- | :--- |
| `CategoricalAttribute` | Finite set of known values, such as booleans, enum values, US states, or status codes. | List of valid categories. | None (domain is already discrete and fixed). |
| `NumericalAttribute` | Ordered integer or continuous floating-point numbers, such as age, balance, or duration. | Inclusive minimum and maximum bounds, plus integer or float data type. | Discretizes into bins via DP quantiles (or uses fixed bin edges if provided). |
| `OpenSetCategoricalAttribute` | Categorical strings whose valid values are unknown in advance, such as free-entered city names or job titles. | None (optional fallback sentinel and optional public seed categories). | Discovers frequent categories via DP Gaussian thresholding (requires positive delta). |
| `FreeFormTextAttribute` | Unstructured text fields synthesized by a language model. | Maximum token length. | Handled by text-generation mechanisms. |

### Out-of-Domain Handling

Real-world datasets frequently contain missing values or records that fall
outside expected bounds. Because raising a runtime error on unexpected records
would leak information about the private data, DPSynth handles out-of-domain
values deterministically during encoding.

For categorical attributes, any value not present in the declared list of
possible values is mapped to a designated fallback index (the first category by
default). If missing or unexpected values may occur, place a typed sentinel
(such as an unknown label for strings or a negative number for integers) at the
first position. If all records are guaranteed to be in-domain, omit the sentinel
so the synthesizer never generates it.

For numerical attributes, values outside the declared minimum and maximum bounds
are clipped to the nearest bound by default, and missing entries map to the
minimum bound. When range clipping is disabled, out-of-range and missing entries
are routed to a dedicated out-of-domain bin during discretization and decoded
back to a sentinel value.

## Specifying Domains

You can define a domain specification directly in the DPSynth domain language,
or derive it automatically from an existing structured container such as a
Pydantic model, a Protocol Buffer descriptor, or Pandas column dtypes. For
Pydantic models and Protocol Buffers, the adapters also provide utilities to
convert record instances to and from a flat tuple representation, making it
straightforward to assemble a flat table for synthesis and reconstruct typed
records from the synthetic output. Because all approaches produce a standard
Python dictionary mapping column names to attribute instances, you can inspect,
serialize, or override individual attributes before calibrating your mechanism.

### 1. Directly in the DPSynth Language

You can construct a domain dictionary directly in Python using the attribute
classes in the domain module:

```python
import dpsynth
from dpsynth import domain

domains = {
    "age": domain.NumericalAttribute(min_value=18, max_value=100, dtype="int"),
    "tier": domain.CategoricalAttribute(possible_values=["FREE", "PRO"]),
    "city": domain.OpenSetCategoricalAttribute(),
}
```

Domain dictionaries can also be saved to and loaded from YAML:

```python
yaml_str = dpsynth.to_yaml(domains)
restored_domains = dpsynth.from_yaml(yaml_str)
```

### 2. From Pydantic Models

Pydantic models can encode both categorical types (booleans, enums, and literal
unions) and numerical bounds directly on the class definition. Optional fields
automatically include a missing-value category or disable numerical range
clipping:

```python
from typing import Literal
from dpsynth.adapters import pydantic as pydantic_adapter
import pydantic


class UserRecord(pydantic.BaseModel):
  age: int = pydantic.Field(ge=18, le=100)
  tier: Literal["FREE", "PRO"]
  city: str


domains = pydantic_adapter.infer_domain(UserRecord)
```

If numeric fields on the Pydantic model do not carry field bound annotations,
you can supply their bounds via the numerical bounds argument. Nested models are
also supported as long as they are non-optional, and are flattened into
dot-separated attribute names.

### 3. From Protocol Buffers

Protocol Buffer descriptors define field types and enum values, but do not store
numeric minimum and maximum ranges:

```protobuf
message UserRecord {
  enum Tier {
    FREE = 0;
    PRO = 1;
  }
  int32 age = 1;
  Tier tier = 2;
  string city = 3;
}
```

Pass your Protobuf message class or descriptor along with numerical bounds:

```python
from dpsynth.adapters import protobuf as protobuf_adapter

domains = protobuf_adapter.infer_domain(
    UserRecord,
    numerical_bounds={"age": (18, 100)},
)
```

Enums and booleans map to categorical attributes, numeric fields map to
numerical attributes using the provided bounds, and strings map to open-set
categorical attributes. Nested submessages are also supported as long as they
are non-repeated (flattened into dot-separated keys), and unsupported or
unbounded fields can be skipped automatically.

### 4. From Pandas Dtypes

When working with typed tabular formats such as Parquet or Arrow, or when a
public schema is already represented as Pandas dtypes, you can infer the domain
from a Series of dtypes or a column-to-dtype mapping:

```python
from dpsynth.adapters import pandas as pandas_adapter
import pandas as pd

schema_dtypes = {
    "age": "int64",
    "tier": pd.CategoricalDtype(["FREE", "PRO"]),
    "city": "string",
}

domains = pandas_adapter.infer_domain(
    schema_dtypes,
    numerical_bounds={"age": (18, 100)},
)
```

```{warning}
When inferring a categorical attribute from a Pandas CategoricalDtype, the list
of categories must come from a public schema. Never cast a sensitive column to a
categorical dtype on private data and pass the resulting dtypes to domain
inference, as doing so extracts the exact active support of the private dataset
without differential privacy. For string columns with unknown categories, leave
the dtype as a string (or object) so it maps to an open-set categorical
attribute.
```

### Customizing Inferred Domains

Because domain inference returns a plain Python dictionary, you can easily
customize specific columns after inference when a schema container does not
capture a nuance of your domain (for example, replacing an inferred open-set
attribute on a string column with a known closed-set categorical attribute, or
specifying custom bin edges on a numerical column):

```python
domains = protobuf_adapter.infer_domain(
    UserRecord,
    numerical_bounds={"age": (18, 100)},
)

# Override an open-set string field when valid values are publicly known:
domains["city"] = domain.CategoricalAttribute(
    possible_values=["OTHER", "NYC", "LA", "SF"],
)
```

## Tabular Schemas vs. Discrete Domains

DPSynth provides two sibling configuration classes that operate at different
levels of data representation depending on whether your dataset is raw tabular
data or already discretized into integers:

| Level | Configuration Class | Domain Representation | Input Data |
| :--- | :--- | :--- | :--- |
| Arbitrary Tabular | `dpsynth.TabularConfig` | `dpsynth.Schema` | Mixed-type tabular data (such as a Pandas DataFrame) |
| Pre-Discretized | `dpsynth.DiscreteConfig` | `mbi.Domain` | Integer-encoded dataset (`mbi.Dataset`) or marginals (`mbi.CliqueVector`) |

For mixed-type datasets containing numbers, known categories, and open-set
strings, pass a schema (or attribute domain dictionary) to a tabular
configuration. During execution, DPSynth automatically allocates a fraction of
the privacy budget to initialize per-column discretizations, encodes the raw
records into integer bin indices, runs the underlying discrete synthesis
mechanism, and decodes the synthetic records back into their original types.

When every attribute $i$ is already encoded as an integer in
$\{0, 1, \dots, k_i - 1\}$, no per-column initialization or type encoding is
needed. Instead of attribute specifications, you define the domain using an MBI
domain that maps each attribute name to its integer cardinality $k_i$, and pass
it to a discrete configuration. If your data is already discretized, going
directly through the discrete mechanism is also slightly more efficient because
it operates over compact integer dtypes with fast NumPy operations, avoiding
the extra initialization and encoding pass of the tabular configuration.

```python
import dpsynth
import mbi

# 1. Arbitrary tabular level: attribute domains bundled in a Schema
tabular_domain = dpsynth.Schema(domains)
tabular_mech = dpsynth.calibrate(
    dpsynth.TabularConfig(), tabular_domain, epsilon=1.0, delta=1e-5
)

# 2. Pre-discretized level: integer cardinalities in an mbi.Domain
discrete_domain = mbi.Domain(["age", "tier", "city"], [32, 2, 50])
discrete_mech = dpsynth.calibrate(
    dpsynth.DiscreteConfig(), discrete_domain, epsilon=1.0, delta=1e-5
)
```
