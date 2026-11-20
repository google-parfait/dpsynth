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

"""Differentially private synthesis for pre-discretized data or marginals.

Demonstrates running a DP discrete synthesis mechanism directly on precomputed
exact marginals (``mbi.CliqueVector``) as well as integer-coded records
(``mbi.Dataset``). Unlike pure post-processing, the mechanism still privately
selects which marginals to measure from the supporting marginals in the
``CliqueVector``. On the full-count 1940 US Census (131.9M rows x 135 columns),
working from the 415 MB precomputed ``CliqueVector`` instead of the 54 GB raw
extract saves substantial pre- and post-processing time and memory.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib

from absl import app
from absl import flags
from absl import logging
import dpsynth
import jax
import mbi
import numpy as np
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for discrete synthesis.

  Attributes:
    input_path: Path to an mbi.CliqueVector or mbi.Dataset .npz archive.
    config_path: Optional YAML path to a serialized DiscreteConfig.
    output_dir: Directory where synthetic data and privacy report are written.
    epsilon: Target differential privacy epsilon.
    delta: Target differential privacy delta.
    seed: Optional random seed for reproducibility.
  """

  input_path: str = 'datasets/census_1940_clique_vector.npz'
  config_path: str | None = None
  output_dir: str = 'dpsynth_output'
  epsilon: float = 1.0
  delta: float = 1e-6
  seed: int | None = None


def main(argv: Sequence[str]) -> None:
  """Calibrates DiscreteConfig and synthesizes discrete records."""
  cfg = simple_parsing.parse(Config, args=argv[1:])
  logging.set_verbosity(logging.INFO)
  jax.config.update('jax_enable_x64', True)

  # DiscreteMechanism accepts precomputed marginals (mbi.CliqueVector) or
  # integer-coded records (mbi.Dataset); mbi.load restores either from .npz.
  data = mbi.load(cfg.input_path)

  # Configure and calibrate the DP discrete synthesis mechanism, restricting
  # candidate selection to supporting marginals when given a CliqueVector.
  workload = data.cliques if isinstance(data, mbi.CliqueVector) else None
  discrete_cfg = dpsynth.DiscreteConfig(
      mechanism=dpsynth.SWIFTConfig(workload=workload),
      use_jax_for_generation=True,
  )
  if cfg.config_path is not None:
    discrete_cfg = dpsynth.from_yaml(cfg.config_path, dpsynth.DiscreteConfig)
  mechanism = dpsynth.calibrate(
      discrete_cfg, epsilon=cfg.epsilon, delta=cfg.delta
  )

  rng = np.random.default_rng(cfg.seed)
  result = mechanism(rng, data)

  # Save the synthetic mbi.Dataset and privacy report to disk.
  out_dir = pathlib.Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_npz = out_dir / 'synthetic_data.npz'
  synth_data = result.synthetic_data
  assert synth_data is not None
  mbi.save(synth_data, out_npz)

  report = dpsynth.PrivacyReport.from_dp_event(mechanism.dp_event)
  dpsynth.to_yaml(report, out_dir / 'privacy_report.yaml')
  logging.info('Wrote %d synthetic rows to %s', synth_data.records, out_npz)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
