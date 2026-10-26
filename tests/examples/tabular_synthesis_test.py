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

"""Tests for tabular_synthesis example binary."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import dpsynth
from examples import tabular_synthesis
from etils import epath
import pandas as pd


class TabularSynthesisTest(parameterized.TestCase):

  def _write_inputs(self, tmp_dir: epath.Path) -> tuple[epath.Path, epath.Path]:
    df = pd.DataFrame({'x': [1, 2, 3, 4] * 25, 'y': ['a', 'b', 'a', 'b'] * 25})
    csv_path = tmp_dir / 'input.csv'
    df.to_csv(csv_path, index=False)
    schema = dpsynth.Schema({
        'x': dpsynth.NumericalAttribute(min_value=0, max_value=10, dtype='int'),
        'y': dpsynth.CategoricalAttribute(['a', 'b']),
    })
    domain_path = tmp_dir / 'domain.yaml'
    dpsynth.to_yaml(schema, domain_path)
    return csv_path, domain_path

  def test_run_default_config(self):
    tmp_dir = epath.Path(self.create_tempdir().full_path)
    csv_path, domain_path = self._write_inputs(tmp_dir)
    out_dir = tmp_dir / 'out'
    cfg = tabular_synthesis.Config(
        input_path=str(csv_path),
        domain_path=str(domain_path),
        output_dir=str(out_dir),
        checkpoint_dir=str(tmp_dir / 'ckpt'),
        seed=42,
    )
    synth_df = tabular_synthesis.run(cfg)
    self.assertEqual(list(synth_df.columns), ['x', 'y'])
    self.assertFalse(synth_df.empty)
    report = dpsynth.from_yaml(
        out_dir / 'privacy_report.yaml', dpsynth.PrivacyReport
    )
    self.assertEqual(report.neighboring_relation, 'ADD_OR_REMOVE_ONE')

  def test_run_with_config_path(self):
    tmp_dir = epath.Path(self.create_tempdir().full_path)
    csv_path, domain_path = self._write_inputs(tmp_dir)
    cfg_yaml = tmp_dir / 'tabular_config.yaml'
    dpsynth.to_yaml(
        dpsynth.TabularConfig(
            discrete_mechanism=dpsynth.discrete_mechanisms.IndependentConfig(),
            numerical_bins=4,
        ),
        cfg_yaml,
    )
    out_dir = tmp_dir / 'out'
    cfg = tabular_synthesis.Config(
        input_path=str(csv_path),
        domain_path=str(domain_path),
        config_path=str(cfg_yaml),
        output_dir=str(out_dir),
        num_rows=30,
        seed=42,
    )
    synth_df = tabular_synthesis.run(cfg)
    self.assertLen(synth_df, 30)
    self.assertTrue((out_dir / 'synthetic_data.csv').exists())


if __name__ == '__main__':
  absltest.main()
