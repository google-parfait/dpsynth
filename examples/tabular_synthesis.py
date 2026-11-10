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
    epsilon: Target differential privacy epsilon.
    delta: Target differential privacy delta.
    seed: Optional random seed for reproducibility.
  """

  input_path: str = 'datasets/adult.csv'
  domain_path: str = 'datasets/adult_domain.yaml'
  config_path: str | None = None
  output_dir: str = 'dpsynth_output'
  epsilon: float = 1.0
  delta: float = 1e-6
  seed: int | None = None


def main(cfg: Config) -> None:
  """Runs calibrated tabular synthesis and writes outputs to disk."""
  logging.set_verbosity(logging.INFO)

  # Configure and calibrate the DP tabular synthesis mechanism.
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

  # Load the private dataset and run the calibrated mechanism.
  rng = np.random.default_rng(cfg.seed)
  if cfg.input_path.endswith('.parquet'):
    df = pd.read_parquet(cfg.input_path)
  else:
    df = pd.read_csv(cfg.input_path)
  result = mechanism(rng, df)

  # Save the synthetic dataset and privacy report to disk.
  out_dir = epath.Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_csv = out_dir / 'synthetic_data.csv'
  synth_df = result.synthetic_data
  synth_df.to_csv(out_csv, index=False)

  report = dpsynth.PrivacyReport.from_dp_event(mechanism.dp_event)
  dpsynth.to_yaml(report, out_dir / 'privacy_report.yaml')
  logging.info('Wrote %d synthetic rows to %s', len(synth_df), out_csv)


if __name__ == '__main__':
  main(simple_parsing.parse(Config))
