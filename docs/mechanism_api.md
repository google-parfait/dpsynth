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

# Mechanism API

This document covers the architectural foundations and design decisions
underlying the DPSynth Mechanism API ({mod}`dpsynth.api`). It is written for
developers extending DPSynth with new synthesis algorithms, integrating
mechanisms into data pipelines, or auditing privacy accounting correctness.

--------------------------------------------------------------------------------

(design-decisions)=
## Design Decisions: `MechanismConfig` vs. `CalibratedMechanism`

DPSynth structures all differential privacy (DP) algorithms around two core
abstractions:

1.  **{class}`~dpsynth.MechanismConfig`**: An abstract base class defining
    the interface for an configuration recipe for a `CalibratedMechanism`.
    Subclasses are generally lightweight frozen dataclasses defined in terms of
    basic python primitives (mechanism hyper-parameters), and can hence be
    serialized and restored via human- and machine-readable formats like YAML.
    This abstract base class defines two key methods: `configure()` and
    `calibrate()`, both of which consume domain information and a privacy
    budget, and return a `CalibratedMechanism`.
2.  **{class}`~dpsynth.CalibratedMechanism`**: An abstract base class
    representing a runnable mechanism with concrete privacy parameters bound.
    The abstract two key methods: `dp_event` and `__call__`. The former provides
    an exact characterization of the mechanisms privacy properties in the
    language of `dp_accounting`, and the latter allows you to run the mechanism
    on actual data. The format of the data can vary between subclasses.

### Architectural Motivation: Decoupling Blueprint from Execution

Earlier prototypes combined hyperparameters, mutable runtime state, privacy
budget allocation, and synthesis execution into single classes. This pattern
created several fundamental issues:

*   **State Mutability & Lifecycle Bugs**: Objects transitioned through
    unconfigured and configured states, where calling execution before
    configuration caused runtime failures or cross-run state contamination.
*   **Nullable Fields**: Privacy parameters (such as noise standard deviations
    $\sigma$ or selection thresholds) were unknown at instantiation, requiring
    fields to be typed as `float | None` and necessitating defensive assertions.
*   **Serialization Ambiguity**: Serializing an algorithm configuration risked
    accidentally capturing transient runtime artifacts or failing on
    non-serializable mathematical state (such as graphical model clique
    vectors).

To resolve this, DPSynth strictly decouples the immutable **recipe**
({class}`~dpsynth.api.MechanismConfig`) from the immutable **runnable instance**
({class}`~dpsynth.api.CalibratedMechanism`):

| Dimension | {class}`~dpsynth.api.MechanismConfig` | {class}`~dpsynth.api.CalibratedMechanism` |
| :--- | :--- | :--- |
| **Role** | Hyperparameter recipe / blueprint | Executable mechanism instance |
| **Mutability** | Frozen dataclass (immutable) | Frozen dataclass (immutable) |
| **Data Dependencies** | None (independent of data & budget) | Bound to domain & concrete budget |
| **Privacy Parameters** | None (holds only budget fractions) | Fully concrete ($\sigma$, thresholds) |
| **Serialization** | Fully serializable to/from YAML | Runtime only (not serialized) |
| **Interface** | `.configure()`, `.calibrate()` | `dp_event`, `__call__(rng, data)` |

By enforcing this separation:

1.  **Immutability**: Once constructed, a
    {class}`~dpsynth.api.CalibratedMechanism` cannot be modified or
    un-calibrated.
2.  **Strict Type Safety**: All parameters on
    {class}`~dpsynth.api.CalibratedMechanism` are non-nullable; no defensive
    assertions are required.
3.  **Clean Separation of Concerns**: Algorithm designers specify *how to
    partition budget* in the config, while the calibrated mechanism focuses
    purely on *how to run* on data.

--------------------------------------------------------------------------------

(three-step-pipeline)=
## The 3-Step Mechanism Pipeline

All DP data generation in DPSynth follows a standardized 3-step lifecycle:

### Step 1: Reusable Configuration

In Step 1, the user or pipeline defines a `MechanismConfig`. This configuration
is completely independent of the dataset size, record values, and privacy
parameters. It can be safely reused across multiple datasets, shared in model
registries, or persisted in configuration files.

```python
import dpsynth

# Step 1: Instantiate a reusable configuration recipe.
config = dpsynth.TabularConfig(
    discrete_mechanism=dpsynth.discrete_mechanisms.AIMConfig(
        pgm_iters=5000,
        max_marginal_size=10_000_000,
        select_budget_fraction=0.5,
    ),
    numerical_bins=32,
    init_budget_fraction=0.1,
)
```

Because `config` contains no private data or privacy parameters, it can be
serialized to disk via `dpsynth.to_yaml(config)`.

### Step 2: Calibrate to Data Domain & Privacy Parameters

In Step 2, the configuration recipe is bound to a specific domain and
target $(\varepsilon, \delta)$-DP budget via
{meth}`~dpsynth.api.MechanismConfig.calibrate`:

1.  **A Domain/Schema**: Specifies attribute classifications (categorical,
    numerical, open-set), value domains, and cross-attribute constraints.
2.  **A Privacy Budget $(\varepsilon, \delta)$**: Target differential privacy
    parameters.

This produces a concrete, runnable {class}`~dpsynth.api.CalibratedMechanism`:

```python
# Define domain schema:
schema = dpsynth.Schema({
    "age": dpsynth.NumericalAttribute(min_value=18, max_value=90),
    "education": dpsynth.CategoricalAttribute(
        categories=["High School", "Bachelors", "Masters", "Doctorate"]
    ),
    "income": dpsynth.NumericalAttribute(min_value=0, max_value=250_000),
})

# Calibrate to target (epsilon, delta)-DP:
calibrated = config.calibrate(schema, epsilon=1.0, delta=1e-5)
```

### Step 3: Run on Sensitive Data

In Step 3, the calibrated mechanism is executed on sensitive data. The
calibrated instance is directly callable ({meth}`~dpsynth.api.CalibratedMechanism.__call__`):

```python
import numpy as np
import pandas as pd

sensitive_df = pd.read_csv("sensitive_data.csv")

rng = np.random.default_rng(seed=42)
result = calibrated(rng, sensitive_df)

synthetic_df = result.synthetic_data
```

--------------------------------------------------------------------------------

(configure-vs-calibrate)=
## `configure(zcdp_rho)` vs. `calibrate(epsilon, delta)`

A common question for developers is: *Why are there two methods for binding a
privacy budget, and how do they interact?*

```python
class MechanismConfig(abc.ABC):

  @abc.abstractmethod
  def configure(
      self, domain=None, *, zcdp_rho: float, delta: float = 0.0
  ) -> CalibratedMechanism:
    """Low-level primitive: map zCDP rho to concrete parameters."""

  def calibrate(
      self, domain=None, *, epsilon: float, delta: float,
  ) -> CalibratedMechanism:
    """High-level entry point: numerical search over rho to satisfy (ε, δ)."""
```

`configure(zcdp_rho)` is the abstract primitive that every `MechanismConfig`
subclass **must** implement. Its responsibilities are:

1.  **Closed-Form Noise Derivation**: Map the scalar $\rho$-zCDP budget directly
    to the mechanisms natural parameters (see {ref}`natural-parameters`). Since
    this mapping is often very simple, it should execute very quickly.
2.  **Deterministic Budget Splitting**: Subdivide $\rho$ across different
    sub-mechanisms (e.g., per-column initialization + base mechanism).

```{important}
The CalibratedMechanism returned by `configure()` should satisfy rho-zCDP, but
this is not the tightest characterization of the privacy properties of the
mechanism! The exact `dp_event` associated with the mechanism can be obtained
via the property `CalibratedMechanism.dp_event`.

**Exercise for the Reader:** Configure the `DirectMechanism` with zcdp_rho=1.0
and compute epsilon for delta=1e-5 using two different methods:

1.  Using the formula $\epsilon = \rho + 2 \cdot \sqrt(\rho \cdot \ln(1/\delta))$,
    or any other zCDP -> DP conversion formula.
2.  Using dp_accounting directly on the `calibrated.dp_event`.

(2) should yield a strictly smaller epsilon than (1). Understanding
this point is critical to understand the design decisions and correctness of the
`configure` and `calibrate` APIs.
```

### `calibrate(epsilon, delta)`:

`calibrate()` is a concrete method defined once on the base
{class}`~dpsynth.api.MechanismConfig` class. It serves users who want to specify
a specific $(\varepsilon, \delta)$ target. The zcdp_rho intermediate
representation is mostly hidden from users of `calibrate()`. Calibrate is
implemented internally using `dp_accouning.calibrate_dp_mechanism`, using
zcdp_rho as the parameter to calibrate and the `dp_event` for what to calibrate
to. Critically, our choice to use zCDP internally for configuration
does not imply any looseness in the final accounting and calibration of our
mechanisms.

--------------------------------------------------------------------------------

(zcdp-intermediate)=
## Notes on the `configure(zcdp_rho)` Design

Using $\rho$-zCDP as the intermediate configuration parameter provides several
key practical advantages:

*   **Single-parameter Calibration**: `dp_accounting.calibrate_dp_mechanism`
    expects an arbitrary function that consumes a single scalar value and
    returns a `DpEvent` object. For simple and homogeneous mechanisms like
    the Gaussian Mechanism, the Exponential Mechanism, or even Poisson-sampled
    DP-SGD, there is usually a single parameter (e.g., $\sigma$, $\varepsilon$,
    or a related quantity) that is natural to calibrate to. In `dpsynth`, our
    mechanisms are heterogeneous compositions of these simpler mechanisms, so
    the natural parameters are multi-dimensional, which `dp_accounting` does
    not know how to calibrate to. We therefore use a single scalar parameter
    to configure to across the entire `dpsynth` API, and have simple functions
    to map this parameter to the natural parameter(s) of the mechanism /
    sub-mechanisms.
*   **Linear Budget Splitting & Arbitrary Nesting**: Unlike
    $(\varepsilon, \delta)$ composition, $\rho$-zCDP composes linearly
    ($\rho = \sum_i \rho_i$). A parent mechanism can divide its budget into
    additive slices ($\rho_i = w_i \cdot \rho$) and pass them down to
    sub-mechanisms cleanly.
*   **Universal Applicability**: All mechanisms used in DPSynth admit a
    well-defined (even if loose) approximate-zCDP guarantee. For example,
    Gaussian queries map directly to $\rho$, while pure $\varepsilon$-DP
    exponential mechanisms satisfy $\rho = \frac{1}{8}\varepsilon^2$-zCDP.
    Crucially, *approximate zCDP at configuration time does not imply loose
    accounting for a calibrated mechanism*: the mechanism's `dp_event`
    is what `calibrate()` evaluates via tight PLD accounting.
*   **Cheap, Closed-Form Parameter Derivation**: Deriving concrete noise
    parameters from $\rho$ requires only simple, closed-form arithmetic (e.g.,
    $\sigma = \sqrt{1 / (2\rho)}$). There are no root-finding loops or
    numerical convolutions inside `configure()`, keeping it fast enough to
    execute hundreds of times per second during binary search calibration.

(natural-parameters)=
## `zcdp_rho` vs. Natural Privacy Parameters

Each DP mechanism has a "natural" parameterization reflecting its mathematical
formulation. However, these natural parameters are heterogeneous and cannot be
composed or split as cleanly:

*   **Gaussian Mechanism**: Natural parameter is the noise standard deviation
    $\sigma$ (or variance $\sigma^2$).
*   **Other Gaussian-DP Mechanisms**: Natural parameter is either $\sigma$ or
    the GDP parameter $\mu^2$ (where $\mu = 1/\sigma$).
*   **Exponential Mechanism**: Natural parameter is $\varepsilon$.
*   **AIM**: Natural parameter is $\rho$.
*   **DP-SGD**: Natural parameters are the tuple `(noise_multiplier,
    sampling_probability, iterations)`.
*   **[DP Quantiles][dp-quantiles-src]**: Natural parameter is a list of step
    budgets $[\varepsilon_1, \dots, \varepsilon_k]$ across recursive
    bisections, whose composite `dp_event` is a composition of exponential
    mechanisms.

[dp-quantiles-src]: https://github.com/google-parfait/dpsynth/blob/main/dpsynth/local_mode/primitives.py

Because these heterogeneous parameters cannot be directly combined,
`configure()` accepts a single scalar `zcdp_rho` and translates it into each
sub-mechanisms natural parameters in closed form. The mechanism then exposes
its exact composition via its {attr}`~dpsynth.api.CalibratedMechanism.dp_event`,
enabling `calibrate()` to provide numerically tight accounting.

## Correctness of `calibrate()` vs. `configure()`

The correctness of `calibrate()` **does not rely on the correctness** of
`configure()`, only that it produces a `CalibratedMechanism` with an honestly
reported `dp_event`. With that being said, `dpsynth` is designed so that
calling `configure()` with a given `zcdp_rho` should produce a
`CalibratedMechanism` has a $\rho$-zCDP guarantee. This makes it safe for users
to call `configure()` directly if they want, although we encourage users to
leverage the higher-level `calibrate()` API since it's tighter.

