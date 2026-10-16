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

"""Public API for DPSynth."""

# pylint: disable=g-importing-member
__version__ = '0.4.0'
from absl import logging
from dpsynth import api
from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth import local_mode
from dpsynth import relational
from dpsynth import reporting
from dpsynth import serialize
from dpsynth._calibration import calibrate
from dpsynth._calibration import with_group_size
from dpsynth._checkpoint import checkpoint
from dpsynth.data_generation_v3 import TabularConfig
from dpsynth.data_generation_v3 import TabularMechanism
from dpsynth.data_generation_v3 import TabularSynthesizer
from dpsynth.discrete_mechanisms.discrete import DiscreteConfig
from dpsynth.discrete_mechanisms.discrete import DiscreteMechanism
from dpsynth.domain import CategoricalAttribute
from dpsynth.domain import FreeFormTextAttribute
from dpsynth.domain import NumericalAttribute
from dpsynth.domain import OpenSetCategoricalAttribute
from dpsynth.domain import Schema
from dpsynth.reporting import PrivacyReport
from dpsynth.serialize import from_yaml
from dpsynth.serialize import to_yaml
import mbi

# Route MBI callback logs (which use print-style formatting) to
# absl.logging.info.
if hasattr(mbi, 'callbacks') and hasattr(mbi.callbacks, 'set_log_fn'):
  mbi.callbacks.set_log_fn(
      lambda *args, sep=' ', **kwargs: logging.info(
          sep.join(str(a) for a in args)
      )
  )

ForeignKeyRelation = relational.ForeignKeyRelation
MultiDataGenerationResult = relational.MultiDataGenerationResult
MultiTableConfig = relational.MultiTableConfig
MultiTableMechanism = relational.MultiTableMechanism

__all__ = [
    'CategoricalAttribute',
    'ForeignKeyRelation',
    'MultiDataGenerationResult',
    'MultiTableConfig',
    'MultiTableMechanism',
    'NumericalAttribute',
    'OpenSetCategoricalAttribute',
    'PrivacyReport',
    'Schema',
    'TabularConfig',
    'TabularMechanism',
    'TabularSynthesizer',
    'api',
    'calibrate',
    'checkpoint',
    'discrete_mechanisms',
    'domain',
    'from_yaml',
    'local_mode',
    'relational',
    'reporting',
    'serialize',
    'to_yaml',
    'with_group_size',
]
