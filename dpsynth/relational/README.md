# `dpsynth.relational` (Multi-Table Relational DP Synthesis)

> [!WARNING]
> **UNDER ACTIVE DEVELOPMENT / EXPERIMENTAL**: This module is in an early
> experimental stage and is under active development. The APIs, internal
> interfaces, and algorithms are subject to breaking changes. It is not yet
> recommended for production use.

---

## Overview

The `dpsynth.relational` package extends DP Synth to support **hierarchical
multi-table relational databases** under differential privacy.

In relational databases (e.g. `Household -> Person -> Activity`), privacy is
protected at the **root parent record** (e.g. the household), while valid
foreign-key linkages and cross-table statistical correlations are preserved
across child tables without generating orphaned records or requiring
intractable flat Cartesian joins.

### Core Approach

- **Cascading Relational Synthesis**: Synthesizes databases table-by-table
  down foreign-key hierarchy trees, ensuring root-entity differential privacy
  without materializing expensive Cartesian joins.
- **Permutation Modeling**: Captures rich parent-child and sibling-to-sibling
  correlations through slot permutation and exchangeability, inspired by
  PrivPetal ([Cai et al., 2025](https://arxiv.org/abs/2503.22970)).
- **Built on `dpsynth` & `mbi` Core**: Reuses existing discrete mechanisms (e.g.
  AIM, MST) and the `mbi` (Private-PGM) graphical model engine under the hood.
- **Unified Mechanism API**: Integrates directly with the `MechanismConfig` /
  `CalibratedMechanism` paradigm used throughout `dpsynth`.

---

## Planned API

```python
import dpsynth
from dpsynth.relational import ForeignKeyRelation, MultiTableConfig

# Define schemas and foreign key relations
foreign_keys = [
    ForeignKeyRelation(
        parent_table="households",
        parent_primary_key="household_id",
        child_table="persons",
        child_foreign_key="household_id",
        max_children_per_parent=5,
    ),
]

# Configure relational synthesizer
config = MultiTableConfig(
    foreign_keys=foreign_keys,
    discrete_mechanism=dpsynth.AIMConfig(),
)

# Calibrate and run
calibrated = config.calibrate(table_domains, epsilon=1.0, delta=1e-5)
result = calibrated(rng, {"households": df_h, "persons": df_p})
synth_tables = result.synthetic_tables
```
