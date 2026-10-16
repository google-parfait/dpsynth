# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## A Note on the Library's Current State

DP Synth is **rapidly evolving** and currently in a malleable state. We have not
yet made a commitment to backwards compatibility, which means **now is the best
time** to propose API changes, suggest new abstractions, or challenge existing
design decisions. If you have opinions about how the library should be
structured, we want to hear them — once we stabilize the API, the bar for
breaking changes will be much higher.

## Types of Contributions

We welcome contributions in many forms:

### Bug Fixes, Issues, and Documentation

-   **Bug reports**: Open an [issue](https://github.com/google/dpsynth/issues)
    describing what went wrong and how to reproduce it.
-   **Bug fixes**: Small, focused PRs that fix a specific issue are always
    appreciated.
-   **Documentation**: Improving docstrings, adding examples, or clarifying
    existing docs.

### Privacy Hardening

Finding and reporting privacy violations is especially valuable. Contributions
that harden existing implementations — for example, integrating with an
[OpenDP](https://opendp.org/) verification backend as an opt-in check — are
welcome.

### New Features and Mechanisms

We accept new mechanisms at every level of the stack:

-   **Tabular data mechanisms**: New algorithms for discrete synthetic data
    generation (e.g., alternatives to AIM, SWIFT, MST).
-   **Text generation mechanisms**: New approaches for differentially private
    text synthesis or fine-tuning.
-   **Primitives**: New building blocks such as partition selection algorithms
    or numerical discretization strategies.
-   **Higher-level components**: Mechanisms that build on top of synthetic data
    generation — for example, adaptive insights from unstructured text, or
    modality-specific pipelines that decompose complex problems into subproblems
    the library already handles well.

### API Design Proposals

The library is at an early stage where API design feedback is highly impactful.
If you think a core abstraction could be improved, open an issue to discuss it
before writing code. We'd rather get the design right now than maintain
backwards-incompatible wrappers later.

### Contributing to Upstream Libraries

DP Synth is co-developed with several companion libraries, and improvements to
any of them directly benefit DP Synth:

-   [**MBI**](https://github.com/ryan112358/mbi): Graphical model estimation and
    inference engine used by the tabular mechanisms.
-   [**dp_accounting**](https://github.com/google/differential-privacy/tree/main/python/dp_accounting):
    Privacy accounting and budget composition.
-   [**jax_privacy**](https://github.com/google-deepmind/jax_privacy):
    Differentially private training primitives used by the text generation
    module.

Contributions to these libraries — bug fixes, performance improvements, new
features — can be surfaced into DP Synth once they land upstream.

## The `contrib/` Directory

If you're proposing a **new mechanism** or a substantial new feature that
doesn't fit neatly into the existing module structure, place it in the
[`contrib/`](contrib/) directory. This is a staging area for experimental or
community-contributed code that hasn't yet been integrated into the core
library.

Once a contribution in `contrib/` has landed, matured, and proven useful, we'll
work with you to migrate it into the appropriate core module.

If your contribution is a bug fix, documentation improvement, or a change to an
existing module, submit it directly to the relevant file — `contrib/` is only
for new standalone additions.

## Acceptance Criteria

When reviewing contributions, we primarily look for two things:

### 1. Clean Mechanism Boundaries and Simplicity

For new mechanisms and subpackages, we look for a clean separation between a
lightweight outer contract and a flexible internal implementation:

-   **Implement `MechanismConfig` / `CalibratedMechanism`**: Conform to the two
    core abstractions in
    [`docs/mechanism_api.md`](docs/mechanism_api.md)
    (`MechanismConfig.configure` / `calibrate` returning a `CalibratedMechanism`
    with `dp_event` and `__call__`). This contract is lightweight and
    modality-agnostic (`domain` and `data` can be whatever types your modality
    requires) while ensuring consistent privacy calibration and accounting
    across the library.
-   **Curate the public API surface**: Be deliberate and thoughtful about what
    is exported in the public API and what the API contract is. Do not expose
    internal helper functions or implementation details to the top-level core
    library.
-   **Flexible internal implementation**: You do not need to use the same
    internal helper functions or data structures that existing mechanisms use.
    Because end users only depend on the curated outer API, your internal
    implementation details and code organization can be completely
    self-contained and change freely over time. Single-file mechanisms (~500
    lines) are welcome for focused algorithms, while larger modality-specific
    mechanisms can live in their own subpackage directory.

### 2. No New Heavy Dependencies

DP Synth deliberately keeps its dependency footprint small:

-   **Do not introduce PyTorch, TensorFlow, or other heavyweight dependencies.**
-   If you need numerical computing, use **JAX**. If you need neural network
    layers, use **Flax**.
-   If your contribution requires a new dependency, discuss it in the issue
    tracker first.

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for managing your development
environment. It provides support for managing `pylock.toml` lock files and is
significantly faster than `pip`.

> [!NOTE] The `pylock.toml` file is generated using `uv`, but it follows
> standard specifications as per PEP 751 and should work with modern package
> managers that support it.

1.  **Install uv:** Follow the
    [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2.  **Create an environment and install dependencies:**

    ```bash
    # Create a virtual environment with Python 3.12
    uv venv --python 3.12
    # Activate it (Linux/macOS)
    source .venv/bin/activate
    # Install all dependencies from the lockfile and install the project in editable mode
    uv pip install -r pylock.toml
    uv pip install -e .
    ```

3.  **Updating `pylock.toml`:** Regenerate `pylock.toml` using `uv` periodically
    or when modifying dependencies in `pyproject.toml`:

    ```bash
    uv export --format pylock.toml --all-extras --default-index https://pypi.org/simple -o pylock.toml --python 3.12
    go run github.com/google/addlicense@v1.2.0 -c "Google LLC" -y "2026" -l apache pylock.toml
    ```

    Commit the updated `pylock.toml` to ensure CI and contributors remain in
    sync.

    Note that any update to `pyproject.toml` must be accompanied by a
    regeneration of `pylock.toml`.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Before opening a pull request, consider
[filing an issue](https://github.com/google/dpsynth/issues) to discuss the
change. This helps us coordinate efforts and provide guidance on the best
approach.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
