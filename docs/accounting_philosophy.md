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

# Privacy Accounting Philosophy

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This page complements the [Mechanism API](mechanism_api.md) guide
by explaining the differential privacy (DP) accounting philosophy in DPSynth:
how the library balances structured mechanism interfaces against functional
primitives, how adaptivity affects composition guarantees, how record-level and
user-level group privacy are modeled, how to inspect fine-grained privacy
profiles via `PrivacyReport`, what is treated as public vs. private across
inputs, outputs, and checkpoints, and how the library handles malformed inputs
and floating-point arithmetic. Note that this document only covers the Local
Mode accounting philosophy; Pipeline Mode uses a different accounting approach
that is not discussed or documented on this page.

--------------------------------------------------------------------------------

(interfaces-vs-flat-functions)=
## Interfaces vs. Primitives

In DPSynth, mechanisms are structured around a two-class abstraction:
`MechanismConfig` and `CalibratedMechanism`. Not every differentially private
routine in DPSynth is wrapped in this class pair; we divide the library into two
layers to balance a unified, composable public API against clean,
low-boilerplate internal code:

*   **Composable mechanisms**: High-level synthesis pipelines (such as
    `TabularMechanism` and `MultiTableMechanism`), discrete marginal mechanisms
    (`Independent`, `Direct`, `MST`, and `AIM`), model trainers, per-column
    domain initializers, and more powerful mechanisms built on top of these
    foundations implement the two-stage interface. Some of these are user-facing
    entry points that can be configured via YAML and calibrated to a target
    $(\varepsilon, \delta)$ via `dpsynth.calibrate`, while others (such as
    per-column domain initializers) are internal sub-mechanisms; rather than
    treating each mechanism as a terminal consumer of a standalone privacy
    budget, the two-stage interface allows them to serve as modular stages in a
    broader pipeline where a higher-level coordinator (like `TabularMechanism`)
    splits the budget across stages and relies on each stage's exact `dp_event`
    for tight composition.
*   **Functional primitives**: Foundational DP building blocks in
    `primitives.py` (the exponential mechanism, Gaussian noise addition,
    histogram quantiles, and Gaussian thresholding) and internal helper routines
    are plain Python functions. At the leaf level, the natural privacy
    parameters (such as $\sigma$ or $\varepsilon$) and pre-aggregated summary
    statistics (histograms or quality scores) are already known; wrapping every
    call in a two-class hierarchy would add boilerplate without aiding
    calibration or composition.

(self-contained-mechanisms)=
### Self-Contained Mechanisms

Individual mechanism implementations in DPSynth (such as each discrete synthesis
mechanism) aim to be self-contained within a single file, delegating only to
established safe building blocks: the foundational DP functions in
`primitives.py`, sub-mechanisms with honestly reported `dp_event`s, and core
data abstractions like `mbi.Dataset` and `mbi.CliqueVector` (where marginal
projection via `.project()` has sensitivity 1).

In some differential privacy codebases, verifying that a mechanism's execution
matches its claimed privacy accounting requires tracing class inheritance
hierarchies, base-class hooks, and DP-critical logic spread across multiple
files. DPSynth does not rely on programmatic enforcement that every touch of the
data is automatically accounted for; instead, it is the responsibility of the
library authors and maintainers to ensure correct usage and accounting of the
different mechanisms. The code is explicitly designed so that auditing
mechanisms for privacy correctness is as simple as it can be, keeping each
mechanism self-contained in a single module with no complex inheritance
behavior.

Privacy correctness on a `CalibratedMechanism` class depends only on `dp_event`
and `__call__()` matching, so a reviewer can open a single file and inspect
those two methods side by side to verify that every data access is accounted for
in the reported privacy event. While this structure does not guarantee the
absence of bugs, it makes discrepancies between noise calibration, sensitivity
scaling, and privacy accounting much easier to spot. For higher-level
mechanisms, this local reasoning is contingent on the assumption that
sub-mechanisms provide correctly reported DP guarantees. For example,
`TabularMechanism` is a lightweight wrapper that composes per-column
initializers with a base discrete mechanism; while you cannot reason about the
end-to-end privacy properties of the mechanism without inspecting all
sub-mechanisms, *conditioned* on the sub-mechanisms being correct, it is easy to
see that `TabularMechanism` accounting is correct.

We are not yet completely uniform in this design split across every corner of
the library today, but pairing functional leaf primitives in `primitives.py`
with self-contained, single-file mechanism implementations is the target state
we are working toward.

--------------------------------------------------------------------------------

(adaptivity)=
## Adaptivity and Composition

Composition theorems in differential privacy depend on how a sequence of
mechanisms $M_1, \dots, M_k$ is chosen:

*   **Non-adaptive and adaptive**: The sequence of mechanisms, step count $k$,
    and privacy parameters are fixed at configuration or calibration time, while
    query choices in $M_i$ may either be fixed upfront (for example, measuring a
    fixed workload of marginals with predetermined Gaussian noise $\sigma$) or
    depend adaptively on the noisy outputs of earlier steps
    $M_1(D), \dots, M_{i-1}(D)$. Most mechanisms in DPSynth fall into this
    category, including recursive quantile bisection
    ([Kaplan et al., 2022](https://arxiv.org/abs/2110.05429)) and MST
    ([McKenna et al., 2021](https://arxiv.org/abs/2108.04978)). Standard Privacy
    Loss Distribution (PLD) accountants
    ([Koskela et al., 2020](https://arxiv.org/abs/1906.03049);
    [Gopi et al., 2021](https://arxiv.org/abs/2106.02848);
    [Doroshenko et al., 2022](https://arxiv.org/abs/2207.04380)) and Rényi DP
    (RDP) accountants ([Mironov, 2017](https://arxiv.org/abs/1702.07476)) remain
    valid because each step's conditional privacy loss is dominated by a
    worst-case distribution known prior to execution
    ([Zhu et al., 2022](https://arxiv.org/abs/2106.08567)).
*   **Fully adaptive**: Both the queries and the step count $k$ or privacy
    parameters ($\sigma_i$, $\varepsilon_i$) are chosen dynamically at runtime
    as a function of earlier noisy outputs
    ([Rogers et al., 2016](https://arxiv.org/abs/1605.08294)). For example, AIM
    ([McKenna et al., 2022](https://arxiv.org/abs/2201.12677)) dynamically
    anneals its per-round budget and round count based on mechanism
    intermediates. Because PLD convolution cannot be precomputed when noise
    scales are data-dependent, general fully adaptive composition at the
    pipeline level is out of scope in DPSynth. Instead, individual fully
    adaptive mechanisms are self-contained: they enforce a budget cap internally
    via a privacy filter and report a single zCDP
    ([Bun & Steinke, 2016](https://arxiv.org/abs/1605.02065);
    [Feldman & Zrnic, 2021](https://arxiv.org/abs/2008.11193)) or GDP
    ([Dong et al., 2022](https://arxiv.org/abs/1905.02383);
    [Smith & Thakurta, 2022](https://arxiv.org/abs/2210.17520);
    [Koskela et al., 2023](https://arxiv.org/abs/2209.15596)) guarantee valid
    under fully adaptive composition, allowing them to be composed alongside
    other stages in a larger pipeline whose stage sequence is fixed upfront.

```{warning}
**Remark (Adaptive Chaining and PLD Accounting):**

Scoping out arbitrary runtime-adaptive composition at the pipeline level allows
DPSynth to use PLD accounting across pipeline stages, which generally provides a
tighter characterization of heterogeneous privacy guarantees than
single-parameter DP definitions like zCDP or pure $\varepsilon$-DP used by
interactive session-based frameworks such as
[OpenDP](https://docs.opendp.org/) and
[Tumult Analytics](https://docs.tmlt.dev/analytics/latest/). Consequently, if
you run one DPSynth mechanism, inspect its output, and decide which mechanism or
privacy parameters to run next based on what you see, you are operating in the
fully adaptive regime. You cannot simply choose privacy parameters on the fly
and compose the two mechanisms' `dp_event` objects post-hoc in a PLD accountant
as if the second mechanism had been fixed upfront. Instead, fix each stage's
privacy budget upfront (or define and calibrate the composed mechanism ahead of
time) before inspecting outputs.
```

--------------------------------------------------------------------------------

(record-vs-user-dp)=
## Neighboring Relation and Group Privacy

*   **Neighboring relation**: All mechanisms in DPSynth are defined under the
    add-or-remove-one neighboring relation, where two datasets $D$ and $D'$ are
    neighbors if $D'$ is obtained from $D$ by adding or removing a single unit.
*   **Record-level default**: By default, the privacy unit is a single record
    (row), assuming each row corresponds to a distinct individual. Every
    calibrated mechanism's `dp_event` reports this base record-level,
    un-subsampled privacy guarantee.
*   **Group privacy**: When a single individual or group may contribute up to
    $k$ records, specifying a group size $k$ in `dpsynth.calibrate` lifts the
    candidate mechanism's record-level privacy guarantee via
    `dpsynth.with_group_size` during calibration (scaling Gaussian and Laplace
    noise multipliers by $1/k$, exponential mechanism $\varepsilon$ by $k$, and
    pure zCDP $\rho$ by $k^2$) so that the calibrated mechanism satisfies
    $(\varepsilon, \delta)$-DP for groups of size $k$. Callers can also use
    `dpsynth.with_group_size` directly to evaluate a mechanism's privacy
    guarantee at any group size $k$.
*   **Caller preconditions**: DPSynth does not perform per-group truncation or
    subsampling internally. When calibrating with a group size $k$ or with
    Poisson subsampling amplification
    ([Balle et al., 2018](https://arxiv.org/abs/1807.01647);
    [Zhu et al., 2022](https://arxiv.org/abs/2106.08567)), the caller must
    enforce the contribution bound or Poisson subsampling during upstream data
    preprocessing.
*   **Poisson subsampling order**: When combining Poisson subsampling with a
    group size $k$, there are two natural preprocessing orders: (1) bound each
    group or individual to at most $k$ records and then Poisson-sample
    individual records, or (2) Poisson-sample *groups* (or individuals) and then
    bound each sampled group to at most $k$ records. DPSynth assumes the second
    model, applying group-size lifting before wrapping the event in an outer
    Poisson-sampled event; record-level Poisson sampling after per-group
    truncation is incompatible with how group privacy lifting is structured.

--------------------------------------------------------------------------------

(privacy-report)=
## Privacy Profiles (`PrivacyReport`)

A differential privacy guarantee is not a single $(\varepsilon, \delta)$ number;
a mechanism induces an entire privacy profile
([Balle et al., 2018](https://arxiv.org/abs/1807.01647)) or trade-off curve
([Dong et al., 2022](https://arxiv.org/abs/1905.02383)) mapping every
$\delta \in (0, 1]$ to a corresponding $\varepsilon(\delta)$. As highlighted by
recent work on DP reporting and attack-aware calibration
([Gomez et al., 2026](https://arxiv.org/abs/2503.10945);
[Kulynych et al., 2024](https://arxiv.org/abs/2407.02191)), reporting only a
single $(\varepsilon, \delta)$ point discards substantial information about a
mechanism's true privacy behavior.

DPSynth provides `dpsynth.PrivacyReport.from_dp_event` to summarize a
mechanism's privacy guarantees across a grid of $(\varepsilon, \delta)$ points,
a hypothesis-testing trade-off curve of $(\text{TPR}, \text{FPR})$ pairs, a
single-parameter $\mu$-GDP estimate, and optional `notes` for recording custom
context or non-DP disclosures:

```python
PrivacyReport(
    dp_event=GaussianDpEvent(noise_multiplier=2.0),
    epsilon_deltas=[
        (2.708, 1e-08),
        (2.490, 1e-07),
        (2.254, 1e-06),
        (1.993, 1e-05),
        (1.698, 1e-04),
    ],
    gdp_estimate=0.5,
    trade_off_curve=TradeOffCurve(
        points=[
            (8.33e-05, 1e-05),
            (0.000643, 0.0001),
            (0.0048, 0.001),
            (0.0339, 0.01),
            (0.217, 0.1),
        ],
        auroc=None,
    ),
    notes=None,
    neighboring_relation='ADD_OR_REMOVE_ONE',
)
```

--------------------------------------------------------------------------------

(public-vs-private)=
## Public vs. Private Boundaries

*   **Public configuration vs. private data**: Everything specified on a
    `MechanismConfig` at configuration or calibration time (including the
    schema, closed categorical vocabularies, numerical bounds, cross-attribute
    constraints, and mechanism hyperparameters) is treated as public knowledge
    and must not be derived from the sensitive dataset without separate privacy
    accounting. Only the dataset passed to a calibrated mechanism at execution
    time is protected by the mechanism's differential privacy guarantee
    (including any data-dependent domain quantities estimated internally during
    execution, such as DP quantile bin edges or open-set vocabularies).
*   **All mechanism outputs are DP**: The entire return value of a calibrated
    mechanism is protected by its reported `dp_event`, with no non-DP side
    outputs. For example, `DataGenerationResult` bundles several artifacts (the
    synthetic DataFrame, the fitted graphical model and noisy measurements, and
    the learned `TabularCodec`), and all of them are jointly covered by the same
    privacy guarantee.
*   **Checkpointing and sensitive intermediates**: When a checkpoint directory
    (via `dpsynth.checkpoint`) or pipeline temporary location is provided during
    execution, mechanisms may materialize non-DP intermediates to disk (such as
    the discretized private dataset or un-noised sufficient statistics) to
    support preemption recovery or distributed execution. It is the caller's
    responsibility to ensure that any checkpoint or temporary directory uses
    appropriately access-controlled, encrypted storage.

--------------------------------------------------------------------------------

(inputs-and-floating-point)=
## Validation and Floating-Point Caveats

*   **Configuration-time validation**: DPSynth validates domains, constraints,
    budget fractions, and parameter compatibility at configuration and
    calibration time before seeing private data. This both prevents runtime
    exceptions on sensitive data from leaking private information and surfaces
    misconfigurations early rather than halfway through a long run. Runtime
    errors during mechanism execution should generally never happen; please
    report an issue if you encounter one.
*   **Malformed records**: During execution, individual cell values (`None`,
    `NaN`, $\pm\infty$, out-of-range numbers, or unknown categories) are
    standardized deterministically per row before any histogram aggregation
    occurs (by clamping to declared bounds or routing to a dedicated
    out-of-domain `<OOD>` bin), ensuring that malformed records do not raise
    data-dependent exceptions that break DP.
*   **Floating-point arithmetic**: Mechanisms use standard IEEE-754
    floating-point arithmetic and continuous PRNGs (`numpy.random.Generator` and
    `jax.random`) rather than exact discrete samplers, so privacy guarantees are
    stated in the idealized real-arithmetic model and do not protect against
    floating-point side channels
    ([Mironov, 2012](https://doi.org/10.1145/2382196.2382264)). Isolating
    low-level primitives in `primitives.py` is intended to simplify adopting
    hardened discrete backends in the future.
