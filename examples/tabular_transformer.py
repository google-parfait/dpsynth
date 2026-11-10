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

"""End-to-end example of TabularTransformer with TabularConfig."""

from __future__ import annotations

import dataclasses

from absl import logging
import dpsynth
import dpsynth.training
from etils import epath
import numpy as np
import pandas as pd
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for TabularTransformer synthesis.

  Attributes:
    input_path: Path to the input CSV dataset.
    domain_path: Path to the domain Schema YAML file.
    output_dir: Directory where synthetic data and privacy report are written.
    epsilon: Target differential privacy epsilon.
    delta: Target differential privacy delta.
    seed: Optional random seed for reproducibility.
  """

  input_path: str = 'datasets/msnbc.csv'
  domain_path: str = 'datasets/msnbc_domain.yaml'
  output_dir: str = 'dpsynth_output'
  epsilon: float = 1.0
  delta: float = 1e-6
  seed: int | None = None


def main(cfg: Config) -> None:
  """Runs calibrated TabularTransformer synthesis and writes outputs to disk."""
  logging.set_verbosity(logging.INFO)

  # Configure and calibrate the DP TabularTransformer mechanism.
  schema = dpsynth.from_yaml(cfg.domain_path)
  transformer_config = dpsynth.training.TabularTransformerConfig(
      num_layers=2,
      n_head=4,
      emb_dim=32,
      num_epochs=5,
      num_iterations=500,
      learning_rate=3e-3,
  )
  tabular_cfg = dpsynth.TabularConfig(
      discrete_mechanism=transformer_config,
      # TabularTransformer does not use 1-way initial measurements, and MSNBC
      # has only closed-domain categorical columns, so reserve almost all of
      # the privacy budget for training the Transformer.
      init_budget_fraction=1e-8,
  )
  mechanism = dpsynth.calibrate(
      tabular_cfg, schema, epsilon=cfg.epsilon, delta=cfg.delta
  )

  # Load the private dataset and run the calibrated mechanism.
  rng = np.random.default_rng(cfg.seed)
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
