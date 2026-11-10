# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example demonstrating Differentially Private synthesis on California Census data.

This script demonstrates how to synthesize multi-table relational data under
Differential Privacy (DP) using DPSynth on the California Census (PUMS) dataset.

Pipeline steps:
1. Load relational schema and domain constraints from domain.yaml.
2. Load parent (household) and child (individual) tables from CSV storage.
3. Configure and calibrate MultiTableConfig with Differential Privacy
parameters.
4. Execute relational synthesis mechanism.
5. Validate relational integrity (primary key uniqueness, foreign key linkage).
6. Evaluate statistical utility (Total Variation Distance) and ML utility
(TSTR).
7. Generate SDMetrics multi-table quality and diagnostic reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Literal

from absl import app
from absl import logging
from dpsynth import discrete_mechanisms
from dpsynth.core import domain
from examples.relational.california_census import example_eval_california_census as eval_utils
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import synthesizer as rel_synth
from etils import epath
import numpy as np
import pandas as pd

# ==============================================================================
# Configuration Constants
# ==============================================================================
DATA_DIR: str = './data/California'
OUTPUT_DIR: str = './data/California/synthetic'


def _find_domain_path() -> epath.Path:
  """Resolves the path to domain.yaml across package and open-source layouts."""
  local_path = epath.Path(__file__).parent / 'domain.yaml'
  if local_path.exists():
    return local_path
  repo_root_path = (
      epath.resource_path('dpsynth').parent
      / 'examples/relational/california_census/domain.yaml'
  )
  if repo_root_path.exists():
    return repo_root_path
  pkg_path = (
      epath.resource_path('dpsynth')
      / 'examples/relational/california_census/domain.yaml'
  )
  if pkg_path.exists():
    return pkg_path
  cwd_path = epath.Path('examples/relational/california_census/domain.yaml')
  if cwd_path.exists():
    return cwd_path
  return local_path


EPSILON: float = 3.2
DELTA: float = 1e-6
NUM_PERMUTATION_SLOTS: int = 3
EXPLORATION_STRATEGY: Literal['empty_token', 'size_sliced'] = 'size_sliced'
RANDOM_SEED: int = 42
PGM_ITERS: int = 100  # Increase for higher fidelity, lower for faster runtime.


# ==============================================================================
# Step 1: Load Domain Schema & Datasets
# ==============================================================================
def load_domain_and_data(
    data_dir: str,
    domain_path: epath.Path | str | None = None,
) -> tuple[
    dict[str, domain.Schema],
    list[rel_domain.ForeignKeyRelation],
    dict[str, pd.DataFrame],
]:
  """Loads relational schema from YAML and California Census tables from storage.

  Args:
    data_dir: Directory containing household.csv and individual.csv.
    domain_path: Path to the relational domain YAML file (defaults to
      auto-detected domain.yaml).

  Returns:
    A tuple of (table_domains, foreign_keys, tables_dict).
  """
  if domain_path is None:
    domain_path = _find_domain_path()
  logging.info('Loading relational domain schema from: %s', domain_path)
  table_domains, foreign_keys = rel_domain.from_yaml_file(str(domain_path))

  logging.info('Loading dataset tables from: %s', data_dir)
  with epath.Path(f'{data_dir}/household.csv').open('r') as f_h:
    household_df = pd.read_csv(f_h)
  with epath.Path(f'{data_dir}/individual.csv').open('r') as f_i:
    individual_df = pd.read_csv(f_i)

  tables = {
      'household': household_df,
      'individual': individual_df,
  }
  logging.info(
      'Loaded %d households and %d individuals.',
      len(tables['household']),
      len(tables['individual']),
  )
  return table_domains, foreign_keys, tables


# ==============================================================================
# Step 2: Configure & Calibrate Multi-Table DP Mechanism
# ==============================================================================
def create_calibrated_mechanism(
    table_domains: Mapping[str, domain.Schema],
    foreign_keys: Sequence[rel_domain.ForeignKeyRelation],
    epsilon: float,
    delta: float,
    num_permutation_slots: int,
    exploration_strategy: Literal['empty_token', 'size_sliced'],
) -> rel_synth.MultiTableMechanism:
  """Configures and calibrates the differential privacy multi-table synthesizer.

  Args:
    table_domains: Dictionary mapping table name to column domain dict.
    foreign_keys: Sequence of ForeignKeyRelation constraints.
    epsilon: Differential privacy epsilon parameter.
    delta: Differential privacy delta parameter.
    num_permutation_slots: Permutation order for exploration.
    exploration_strategy: 'empty_token' or 'size_sliced'.

  Returns:
    A calibrated MultiTableMechanism ready to synthesize data.
  """
  logging.info(
      'Configuring MultiTableConfig with epsilon=%.2f, delta=%e, slots=%d,'
      ' strategy=%s',
      epsilon,
      delta,
      num_permutation_slots,
      exploration_strategy,
  )
  config = rel_synth.MultiTableConfig(
      foreign_keys=foreign_keys,
      discrete_mechanism=discrete_mechanisms.MSTConfig(pgm_iters=PGM_ITERS),
      num_permutation_slots=num_permutation_slots,
      exploration_strategy=exploration_strategy,
  )
  mechanism = config.calibrate(table_domains, epsilon=epsilon, delta=delta)
  assert isinstance(mechanism, rel_synth.MultiTableMechanism)
  logging.info('Mechanism calibrated successfully.')
  return mechanism


# ==============================================================================
# Step 3: Save Synthetic Data to Storage
# ==============================================================================
def save_synthetic_tables(
    synthetic_tables: Mapping[str, pd.DataFrame],
    output_dir: str,
) -> None:
  """Saves synthetic tables as CSV files to the specified output directory.

  Args:
    synthetic_tables: Mapping of table names to synthesized DataFrames.
    output_dir: Output directory path on disk or local storage.
  """
  logging.info('Saving synthetic tables to: %s', output_dir)
  output_path = epath.Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  for table_name, df in synthetic_tables.items():
    file_path = output_path / f'{table_name}.csv'
    logging.info('Writing %s (%d rows) to %s', table_name, len(df), file_path)
    with file_path.open('w') as f:
      df.to_csv(f, index=False)
  logging.info('Synthetic tables successfully saved.')


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('=== Relational DPSynth: California Census Example ===')

  # 1. Load schema and real data
  table_domains, foreign_keys, real_tables = load_domain_and_data(
      data_dir=DATA_DIR,
  )

  # 2. Configure and calibrate DP mechanism
  mechanism = create_calibrated_mechanism(
      table_domains=table_domains,
      foreign_keys=foreign_keys,
      epsilon=EPSILON,
      delta=DELTA,
      num_permutation_slots=NUM_PERMUTATION_SLOTS,
      exploration_strategy=EXPLORATION_STRATEGY,
  )

  # 3. Execute DP synthesis
  logging.info('Starting relational synthesis...')
  rng = np.random.default_rng(RANDOM_SEED)
  result = mechanism(rng=rng, data=real_tables)
  synthetic_tables = dict(result.synthetic_tables)
  logging.info(
      'Synthesis complete: %d households, %d individuals.',
      len(synthetic_tables['household']),
      len(synthetic_tables['individual']),
  )

  # 4. Save synthetic datasets to storage
  save_synthetic_tables(synthetic_tables, output_dir=OUTPUT_DIR)

  # 5. Validate relational integrity (PK uniqueness, 0 orphans, max capacity)
  eval_utils.validate_relational_integrity(synthetic_tables)

  # 6. Statistical & ML utility evaluations
  # eval_utils.evaluate_statistical_fidelity(real_tables, synthetic_tables)
  # eval_utils.evaluate_downstream_ml_utility(
  #    real_tables, synthetic_tables, random_state=RANDOM_SEED
  # )

  # 7. SDMetrics multi-table diagnostic and quality reports
  eval_utils.generate_sdmetrics_reports(real_tables, synthetic_tables)
  logging.info('=== Relational DPSynth Pipeline Completed Successfully ===')
  os._exit(0)  # pylint: disable=protected-access


if __name__ == '__main__':
  app.run(main)
