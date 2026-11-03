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

  input_path: str = 'datasets/adult.csv'
  domain_path: str = 'datasets/adult_domain.yaml'
  output_dir: str = 'dpsynth_output'
  epsilon: float = 1.0
  delta: float = 1e-6
  seed: int | None = None


def run(cfg: Config) -> pd.DataFrame:
  """Runs calibrated TabularTransformer synthesis and writes outputs to disk."""
  rng = np.random.default_rng(cfg.seed)
  df = pd.read_csv(cfg.input_path)
  schema = dpsynth.from_yaml(cfg.domain_path)

  transformer_config = dpsynth.training.TabularTransformerConfig(
      num_layers=2,
      n_head=2,
      emb_dim=16,
      num_epochs=100,
      num_iterations=1000,
      learning_rate=1e-3,
  )
  tabular_cfg = dpsynth.TabularConfig(
      discrete_mechanism=transformer_config,
      numerical_bins=8,
  )

  mechanism = dpsynth.calibrate(
      tabular_cfg, schema, epsilon=cfg.epsilon, delta=cfg.delta
  )
  result = mechanism(rng, df)

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
