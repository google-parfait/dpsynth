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

"""End-to-end example of TabularTransformer with TabularConfig.

This example loads a tabular dataset and its domain specification,
configures a TabularConfig with a TabularTransformerConfig discrete mechanism,
and runs the synthesizer. TabularConfig handles all discretization and
encoding/decoding automatically.
"""

import dataclasses

import dpsynth
from dpsynth.experimental.synthetic_transformer import synth_transformer
from etils import epath
import numpy as np
import pandas as pd
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for TabularTransformer synthesis.

  Attributes:
    csv_path: Path to input dataset CSV file.
    domain_path: Path to dataset domain YAML specification.
    seed: Optional random seed for reproducibility.
  """

  csv_path: str = "datasets/adult.csv"
  domain_path: str = "datasets/adult_domain.yaml"
  seed: int | None = None


def main(cfg: Config) -> None:
  # 1. LOAD DATA
  csv_path = cfg.csv_path
  print(f"Loading dataset from {csv_path}...")
  with epath.Path(csv_path).open("r") as f:
    df = pd.read_csv(f)

  print(f"Loaded dataset with shape: {df.shape}")

  # 2. LOAD DOMAIN
  domain_path = cfg.domain_path
  print(f"\nLoading domain from {domain_path}...")
  attribute_domains = dpsynth.from_yaml(str(domain_path))

  # Use columns from the dataset that exist in the domain
  domain_spec = {
      col: attribute_domains[col]
      for col in df.columns
      if col in attribute_domains
  }
  df = df[list(domain_spec.keys())]

  transformer_config = synth_transformer.TabularTransformerConfig(
      num_layers=2,
      n_head=2,
      emb_dim=16,
      num_epochs=100,
      num_iterations=1000,
      learning_rate=1e-3,
  )

  # 4. INITIALIZE AND CALIBRATE TABULAR SYNTHESIZER
  print("\nInitializing and calibrating TabularConfig...")
  synth = dpsynth.TabularConfig(
      discrete_mechanism=transformer_config,
      numerical_bins=8,
  )
  # Configure with budget = np.inf for non-DP mode.
  calibrated_synth = synth.configure(domain_spec, budget=np.inf)

  # 5. RUN SYNTHESIZER
  print(
      "Running synthesizer (this will discretize, train, sample, and decode)..."
  )
  rng = np.random.default_rng(cfg.seed)
  result = calibrated_synth(rng, df)
  df_synth = result.synthetic_data

  print("\nSample synthetic data:")
  print(df_synth.head(5))

  # 6. COMPARE STATS (Quality Check)
  print("\n--- Comparing Statistics (Quality Check) ---")

  # Categorical check
  cat_cols = [
      col
      for col, spec in domain_spec.items()
      if isinstance(spec, dpsynth.CategoricalAttribute)
  ]
  if cat_cols:
    sample_cat = cat_cols[0]
    print(f"\nMarginal distribution for '{sample_cat}' (categorical):")
    orig_dist = df[sample_cat].value_counts(normalize=True)
    synth_dist = df_synth[sample_cat].value_counts(normalize=True)
    print(
        pd.DataFrame({"Original": orig_dist, "Synthetic": synth_dist}).fillna(0)
    )

  # Numerical check
  num_cols = [
      col
      for col, spec in domain_spec.items()
      if isinstance(spec, dpsynth.NumericalAttribute)
  ]
  if num_cols:
    print("\nMean values for numerical columns:")
    orig_means = df[num_cols].mean()
    synth_means = df_synth[num_cols].mean()
    print(
        pd.DataFrame(
            {"Original Mean": orig_means, "Synthetic Mean": synth_means}
        )
    )


if __name__ == "__main__":
  main(simple_parsing.parse(Config))
