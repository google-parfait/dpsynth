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

"""Library for input/output transformations for data generation pipelines."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
import csv
import glob
import os
import pickle
from typing import TYPE_CHECKING, Any

import apache_beam as beam
from dpsynth.pipeline_transformations import types
import pandas as pd

if TYPE_CHECKING:
  import tensorflow as tf

def load_csv(pipeline: beam.Pipeline, path: str) -> beam.PCollection:
  """Loads the data to generate synthetic data for."""
  all_files = glob.glob(path)
  if not all_files:
    raise ValueError(f'No files found matching path {path}')

  return (
      pipeline
      | 'Load filenames' >> beam.Create(all_files)
      | 'Read CSVs'
      >> beam.FlatMap(lambda file_path: pd.read_csv(file_path).iterrows())
  )


def save_csv(
    data: Iterable[tuple[Any, ...]],
    path: str,
    attributes: tuple[str, ...],
):
  """Saves the synthetic data to a CSV file."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(attributes)
    writer.writerows(data)


def load_data_local(
    path: str, data_format: types.DataFormat
) -> list[Mapping[str, Any]]:
  """Loads the data for local data generation."""
  match data_format:
    case types.DataFormat.CSV:
      all_files = glob.glob(path)
      if not all_files:
        raise ValueError(f'No files found matching path {path}')
      return pd.concat([pd.read_csv(file) for file in all_files]).iterrows()  # pyrefly: ignore[bad-return]
    case _:
      raise ValueError(f'Unsupported data format: {data_format}')


def save_data_local(
    path: str,
    data: Iterable[tuple[Any, ...] | tf.train.Example],
    data_format: types.DataFormat,
    attributes: tuple[str, ...],
):
  """Saves the synthetic data to a file locally."""
  match data_format:
    case types.DataFormat.CSV:
      save_csv(data, path, attributes)
    case types.DataFormat.TFRECORD:
      import tensorflow as tf
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with tf.io.TFRecordWriter(path) as writer:
        for record in data:
          # record is expected to be a proto message.
          writer.write(record.SerializeToString())
    case _:
      raise ValueError(f'Unsupported data format: {data_format}')


def save_model_pipeline(
    model_collection: beam.PCollection,
    descriptor_collection: beam.PCollection,
    path: str,
) -> None:
  """Saves PCollection of model and descriptor to the filesystem (Pickle via TFRecord)."""

  def serialize(model, descriptor):
    return pickle.dumps((model, descriptor))

  serialized = model_collection | 'SerializeModel' >> beam.Map(
      serialize, descriptor=beam.pvalue.AsSingleton(descriptor_collection)
  )
  _ = serialized | 'WriteModel' >> beam.io.WriteToTFRecord(
      path, coder=beam.coders.BytesCoder()
  )


def load_model_pipeline(
    pipeline: beam.Pipeline,
    path: str,
) -> tuple[beam.PCollection, beam.PCollection]:
  """Loads PCollection of model and descriptor from the filesystem."""
  if not path.endswith('*'):
    path += '*'
  serialized = pipeline | 'ReadModel' >> beam.io.ReadFromTFRecord(
      path, coder=beam.coders.BytesCoder()
  )

  def deserialize(data):
    return pickle.loads(data)

  unpickled = serialized | 'DeserializeModel' >> beam.Map(deserialize)
  models = unpickled | 'ExtractModel' >> beam.Map(lambda x: x[0])
  descriptors = unpickled | 'ExtractDescriptor' >> beam.Map(lambda x: x[1])
  return models, descriptors


def save_model_local(path: str, model: Any, descriptor: Any) -> None:
  """Saves model and descriptor locally using Pickle."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as f:
    pickle.dump((model, descriptor), f)


def load_model_local(path: str) -> tuple[Any, Any]:
  """Loads model and descriptor locally using Pickle."""
  with open(path, 'rb') as f:
    return pickle.load(f)


def save_diagnostic_info_pipeline(
    diagnostic_info_collection: beam.PCollection,
    path: str,
) -> None:
  """Saves PCollection of diagnostic info to a text file."""
  _ = (
      diagnostic_info_collection
      | 'ToTextProto' >> beam.Map(str)
      | 'WriteDiagnostics' >> beam.io.WriteToText(path, shard_name_template='')
  )


def save_diagnostic_info_local(path: str, diagnostic_info: Any) -> None:
  """Saves diagnostic info locally as a text file."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    f.write(str(diagnostic_info))


def load_data_for_beam(
    pipeline: beam.Pipeline,
    path: str,
    data_format: types.DataFormat,
) -> beam.PCollection:
  """Loads the data to generate synthetic data for using Beam."""
  match data_format:
    case types.DataFormat.CSV:
      return load_csv(pipeline, path)
    case types.DataFormat.TFRECORD:
      import tensorflow as tf
      return pipeline | 'ReadTFRecord' >> beam.io.ReadFromTFRecord(
          path, coder=beam.coders.ProtoCoder(tf.train.Example)
      )
    case _:
      raise ValueError(f'Unsupported data format for Beam: {data_format}')


def save_beam_data(
    data: beam.PCollection,
    path: str,
    data_format: types.DataFormat,
    attributes: tuple[str, ...],
) -> None:
  """Saves the synthetic data to a file(s) using Beam."""
  match data_format:
    case types.DataFormat.TFRECORD:
      import tensorflow as tf
      _ = data | 'WriteTFRecord' >> beam.io.WriteToTFRecord(
          path, coder=beam.coders.ProtoCoder(tf.train.Example)
      )
    case types.DataFormat.CSV:

      def format_to_csv(row: tuple[Any, ...] | None) -> str:
        return ','.join(map(str, row)) if row else ''

      _ = (
          data
          | 'FormatToCSV' >> beam.Map(format_to_csv)
          | 'WriteCSV'
          >> beam.io.WriteToText(path, header=format_to_csv(attributes))
      )
    case _:
      raise ValueError(f'Unsupported data format for Beam: {data_format}')
