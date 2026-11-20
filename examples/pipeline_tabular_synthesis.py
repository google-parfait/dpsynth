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

"""Distributed differentially private tabular synthesis with Apache Beam.

Demonstrates scaling tabular synthesis beyond single-machine memory using the
``dpsynth.pipeline`` API backed by PipelineDP. Instead of loading the entire
table into a Pandas DataFrame as in ``tabular_synthesis.py``, records stream
through an Apache Beam pipeline (or an in-process Python iterable via
``--no-use_beam`` for quick local debugging) to compute differentially private
marginals and emit synthetic CSV rows (using the UCI Adult dataset by default).
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
import dataclasses
import io

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import dpsynth
import dpsynth.pipeline
from etils import epath
import pandas as pd
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Config:
  """Command-line configuration for PipelineDP tabular synthesis.

  Attributes:
    input_path: Path to the input CSV dataset.
    domain_path: Optional path to the domain Schema YAML file.
    output_dir: Directory where synthetic data is written.
    mechanism: Marginal-based synthesis mechanism to run.
    epsilon: Target differential privacy epsilon.
    delta: Target differential privacy delta.
    use_beam: Whether to execute via Apache Beam rather than in-process.
  """

  input_path: str = 'datasets/adult.csv'
  domain_path: str | None = 'datasets/adult_domain.yaml'
  output_dir: str = 'dpsynth_output'
  mechanism: dpsynth.pipeline.Mechanism = dpsynth.pipeline.Mechanism.MST
  epsilon: float = 1.0
  delta: float = 1e-6
  use_beam: bool = True


def _to_csv_line(row) -> str:
  """Formats a record tuple as an RFC 4180 CSV line."""
  buf = io.StringIO()
  csv.writer(buf, lineterminator='').writerow(row)
  return buf.getvalue()


def main(argv: Sequence[str]) -> None:
  """Runs PipelineDP tabular synthesis and writes synthetic CSV data to disk."""
  cfg = simple_parsing.parse(Config, args=argv[1:])
  logging.set_verbosity(logging.INFO)

  # Build the CSV dataset descriptor and apply any optional domain schema.
  sample_df = pd.read_csv(cfg.input_path, nrows=100)
  descriptor = dpsynth.pipeline.get_dataset_descriptor_for_csv(sample_df)
  if cfg.domain_path is not None:
    domains = dpsynth.from_yaml(cfg.domain_path)
    descriptor.update_from_domain_specification(domains)

  gen_cfg = dpsynth.pipeline.DataGenerationConfig(
      epsilon=cfg.epsilon,
      delta=cfg.delta,
      mechanism=cfg.mechanism,
      dataset_descriptor=descriptor,
      data_format=dpsynth.pipeline.DataFormat.CSV,
  )

  # Execute the pipeline on Apache Beam or in-process and save synthetic CSV.
  out_dir = epath.Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_csv = out_dir / 'synthetic_data.csv'

  if cfg.use_beam:
    header = _to_csv_line(descriptor.attribute_names)
    with beam.Pipeline() as beam_pipeline:
      raw_rows = (
          beam_pipeline
          | 'CreatePath' >> beam.Create([cfg.input_path])
          | 'ReadCSV' >> beam.FlatMap(lambda p: pd.read_csv(p).iterrows())
      )
      synth_rows = dpsynth.pipeline.generate(raw_rows, gen_cfg)
      _ = (
          synth_rows
          | 'FormatCSV' >> beam.Map(_to_csv_line)
          | 'WriteCSV'
          >> beam.io.WriteToText(
              str(out_csv), header=header, shard_name_template=''
          )
      )
  else:
    rows = list(pd.read_csv(cfg.input_path).iterrows())
    synth_rows = list(dpsynth.pipeline.generate(rows, gen_cfg))
    columns = list(descriptor.attribute_names)
    synth_df = pd.DataFrame(synth_rows, columns=columns)
    synth_df.to_csv(out_csv, index=False)

  logging.info('Wrote synthetic data to %s', out_csv)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
