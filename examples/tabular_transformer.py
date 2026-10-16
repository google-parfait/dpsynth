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

"""End-to-end example of SyntheticTransformer with TabularSynthesizer.

This example loads a tabular dataset and its domain specification,
configures a TabularSynthesizer with a custom TransformerDiscreteMechanism
(which wraps the SyntheticTransformer), and runs the synthesizer.
TabularSynthesizer handles all discretization and encoding/decoding
automatically.
"""

from absl import app
from absl import flags
from dpsynth import data_generation_v3
from dpsynth import domain
from dpsynth.experimental.synthetic_transformer import synth_transformer
from etils import epath
import numpy as np
import pandas as pd

_CSV_PATH = flags.DEFINE_string(
    "csv_path",
    None,
    "Path to input dataset CSV file.",
    required=True,
)
_DOMAIN_PATH = flags.DEFINE_string(
    "domain_path",
    None,
    "Path to dataset domain YAML specification.",
    required=True,
)


def main(argv):
  del argv

  # 1. LOAD DATA
  csv_path = _CSV_PATH.value
  print(f"Loading dataset from {csv_path}...")
  with epath.Path(csv_path).open("r") as f:
    df = pd.read_csv(f)

  print(f"Loaded dataset with shape: {df.shape}")

  # 2. LOAD DOMAIN
  domain_path = _DOMAIN_PATH.value
  print(f"\nLoading domain from {domain_path}...")
  attribute_domains = domain.from_yaml_file(str(domain_path))

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
  )
  discrete_mechanism = synth_transformer.TabularTransformer(
      config=transformer_config,
      num_epochs=100,
      num_iterations=1000,
      learning_rate=1e-3,
  )

  # 4. INITIALIZE AND CALIBRATE TABULAR SYNTHESIZER
  print("\nInitializing and calibrating TabularConfig...")
  synth = data_generation_v3.TabularConfig(
      # pyrefly: ignore[bad-argument-type]
      discrete_mechanism=discrete_mechanism,
      numerical_bins=8,
  )
  # Configure with budget = np.inf for non-DP mode.
  calibrated_synth = synth.configure(domain_spec, budget=np.inf)

  # 5. RUN SYNTHESIZER
  print(
      "Running synthesizer (this will discretize, train, sample, and decode)..."
  )
  rng = np.random.default_rng(42)
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
      if isinstance(spec, domain.CategoricalAttribute)
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
      if isinstance(spec, domain.NumericalAttribute)
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
  app.run(main)
