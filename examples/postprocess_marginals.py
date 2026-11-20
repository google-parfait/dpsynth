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

"""Post-processing differentially private marginals into synthetic data.

Demonstrates synthesizing a consistent tabular dataset purely by
post-processing noisy DP marginal tables that were already computed and
privatized (for example, by an external query engine or distributed SQL
pipeline). Unlike ``discrete_synthesis.py``, query selection and noise addition
happen upstream, so fitting the graphical model with
``UniversalAcceleratedMethod`` and sampling records consumes no additional
privacy budget and requires no access to the 54 GB raw 1940 US Census extract
(131.9M rows x 135 columns), operating instead on 2.2 MB of marginal CSVs.
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
import pandas as pd
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for marginal post-processing.

  Noise is added inline here only to keep the example self-contained (default
  sigma=50 corresponds to roughly epsilon=0.91, delta=1e-6 across the 114
  Census tables). Because this workflow sits outside DPSynth's Mechanism API,
  in practice the marginals loaded from disk would already be privatized by an
  upstream system.

  Attributes:
    marginals_dir: Directory containing marginal CSV files.
    output_dir: Directory where synthetic data is written.
    sigma: Gaussian noise standard deviation added to each marginal cell.
    iters: Maximum number of graphical model optimization iterations.
    num_rows: Number of synthetic rows to sample, or None to match total count.
    seed: Optional random seed for reproducibility.
  """

  marginals_dir: str = 'datasets/census_marginals'
  output_dir: str = 'dpsynth_output'
  sigma: float = 50.0
  iters: int = 10_000
  num_rows: int | None = 10_000
  seed: int | None = None


def main(argv: Sequence[str]) -> None:
  """Adds DP noise to marginals, fits an MRF with UAM, and samples CSV data."""
  cfg = simple_parsing.parse(Config, args=argv[1:])
  logging.set_verbosity(logging.INFO)
  jax.config.update('jax_enable_x64', True)

  # Load exact marginal CSV tables from the input directory.
  marginal_paths = sorted(pathlib.Path(cfg.marginals_dir).glob('*.csv'))
  if not marginal_paths:
    raise ValueError(f'No marginal CSV files found in {cfg.marginals_dir}.')
  exact_marginals = [pd.read_csv(p) for p in marginal_paths]
  num_tables = len(exact_marginals)
  logging.info('Loaded %d marginals from %s', num_tables, cfg.marginals_dir)

  # Add Gaussian noise to simulate DP marginals.
  rng = np.random.default_rng(cfg.seed)
  noisy_marginals = [
      df.assign(count=df['count'] + rng.normal(0.0, cfg.sigma, len(df)))
      for df in exact_marginals
  ]

  # Fit a consistent graphical model with UAM and sample synthetic records.
  synth_df = dpsynth.generate_synthetic_data_from_marginals(
      noisy_marginals,
      iters=cfg.iters,
      log=True,
      nrows=cfg.num_rows,
      estimator=mbi.estimation.UniversalAcceleratedMethod(),
  )

  # Save the synthetic dataset to disk.
  out_dir = pathlib.Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_csv = out_dir / 'synthetic_data.csv'
  synth_df.to_csv(out_csv, index=False)
  logging.info('Wrote %d synthetic rows to %s', len(synth_df), out_csv)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
