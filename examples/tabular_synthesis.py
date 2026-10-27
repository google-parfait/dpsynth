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

"""In-memory differentially private tabular synthesis example."""

from __future__ import annotations

import dataclasses

from absl import logging
import dpsynth
from etils import epath
import numpy as np
import pandas as pd
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for in-memory tabular synthesis.

  Attributes:
    input_path: Path to the input CSV or Parquet dataset.
    domain_path: Path to the domain Schema YAML file.
    config_path: Optional YAML path to a serialized TabularConfig.
    output_dir: Directory where synthetic data and privacy report are written.
    checkpoint_dir: Optional directory for intermediate stage checkpointing.
    epsilon: Target differential privacy epsilon.
    delta: Target differential privacy delta.
    num_rows: Optional number of synthetic rows to generate.
    seed: Optional random seed for reproducibility.
  """

  input_path: str = 'datasets/adult.csv'
  domain_path: str = 'datasets/adult_domain.yaml'
  config_path: str | None = None
  output_dir: str = 'dpsynth_output'
  checkpoint_dir: str | None = None
  epsilon: float = 1.0
  delta: float = 1e-6
  num_rows: int | None = None
  seed: int | None = None


def _read_table(path: str) -> pd.DataFrame:
  """Reads a CSV or Parquet table from disk."""
  if path.endswith('.parquet'):
    return pd.read_parquet(path)
  return pd.read_csv(path)


def run(cfg: Config) -> pd.DataFrame:
  """Runs calibrated tabular synthesis and writes outputs to disk."""
  rng = np.random.default_rng(cfg.seed)
  df = _read_table(cfg.input_path)
  schema = dpsynth.from_yaml(cfg.domain_path)

  if cfg.config_path is not None:
    tabular_cfg = dpsynth.from_yaml(cfg.config_path, dpsynth.TabularConfig)
  else:
    tabular_cfg = dpsynth.TabularConfig(
        discrete_mechanism=dpsynth.discrete_mechanisms.SWIFTConfig(),
        numerical_bins=32,
    )

  mechanism = dpsynth.calibrate(
      tabular_cfg, schema, epsilon=cfg.epsilon, delta=cfg.delta
  )
  with dpsynth.checkpoint(cfg.checkpoint_dir):
    result = mechanism(rng, df, num_rows=cfg.num_rows)

  out_dir = epath.Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_csv = out_dir / 'synthetic_data.csv'
  result.synthetic_data.to_csv(out_csv, index=False)

  report = dpsynth.PrivacyReport.from_dp_event(
      mechanism.dp_event, target_deltas=[cfg.delta]
  )
  dpsynth.to_yaml(report, out_dir / 'privacy_report.yaml')
  logging.info(
      'Wrote %d synthetic rows to %s', len(result.synthetic_data), out_csv
  )
  return result.synthetic_data


def main() -> None:
  logging.set_verbosity(logging.INFO)
  run(simple_parsing.parse(Config))


if __name__ == '__main__':
  main()
