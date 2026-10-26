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
from dpsynth import postprocessing
from dpsynth import relational
from dpsynth import reporting
from dpsynth import serialize
from dpsynth._calibration import calibrate
from dpsynth._calibration import with_group_size
from dpsynth._checkpoint import checkpoint
from dpsynth.api import CalibratedMechanism
from dpsynth.api import MechanismConfig
from dpsynth.constraints import Constraint
from dpsynth.data_generation_v3 import ColumnCodec
from dpsynth.data_generation_v3 import create_initializers
from dpsynth.data_generation_v3 import DataGenerationResult
from dpsynth.data_generation_v3 import TabularCodec
from dpsynth.data_generation_v3 import TabularConfig
from dpsynth.data_generation_v3 import TabularMechanism
from dpsynth.data_generation_v3 import TabularSynthesizer
from dpsynth.discrete_mechanisms.aim import AIM
from dpsynth.discrete_mechanisms.aim import AIMConfig
from dpsynth.discrete_mechanisms.common import DiscreteMechanismResult
from dpsynth.discrete_mechanisms.common import MechanismDiagnostics
from dpsynth.discrete_mechanisms.direct import Direct
from dpsynth.discrete_mechanisms.direct import DirectConfig
from dpsynth.discrete_mechanisms.discrete import DiscreteConfig
from dpsynth.discrete_mechanisms.discrete import DiscreteMechanism
from dpsynth.discrete_mechanisms.independent import Independent
from dpsynth.discrete_mechanisms.independent import IndependentConfig
from dpsynth.discrete_mechanisms.mst import MST
from dpsynth.discrete_mechanisms.mst import MSTConfig
from dpsynth.discrete_mechanisms.swift import SWIFT
from dpsynth.discrete_mechanisms.swift import SWIFTConfig
from dpsynth.domain import AttributeType
from dpsynth.domain import CategoricalAttribute
from dpsynth.domain import FreeFormTextAttribute
from dpsynth.domain import NumericalAttribute
from dpsynth.domain import OpenSetCategoricalAttribute
from dpsynth.domain import Schema
from dpsynth.local_mode.initialization import CategoricalInitializer
from dpsynth.local_mode.initialization import CategoricalInitializerConfig
from dpsynth.local_mode.initialization import CategoricalMeasurement
from dpsynth.local_mode.initialization import ColumnMeasurement
from dpsynth.local_mode.initialization import NumericalInitializer
from dpsynth.local_mode.initialization import NumericalInitializerConfig
from dpsynth.local_mode.initialization import NumericalMeasurement
from dpsynth.local_mode.initialization import OpenSetInitializer
from dpsynth.local_mode.initialization import OpenSetInitializerConfig
from dpsynth.local_mode.initialization import OpenSetMeasurement
from dpsynth.postprocessing import encode_marginals
from dpsynth.postprocessing import generate_synthetic_data_from_marginals
from dpsynth.postprocessing import infer_categorical_domain
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
    'AIM',
    'AIMConfig',
    'AttributeType',
    'CalibratedMechanism',
    'CategoricalAttribute',
    'CategoricalInitializer',
    'CategoricalInitializerConfig',
    'CategoricalMeasurement',
    'ColumnCodec',
    'ColumnMeasurement',
    'Constraint',
    'DataGenerationResult',
    'Direct',
    'DirectConfig',
    'DiscreteConfig',
    'DiscreteMechanism',
    'DiscreteMechanismResult',
    'ForeignKeyRelation',
    'FreeFormTextAttribute',
    'Independent',
    'IndependentConfig',
    'MST',
    'MSTConfig',
    'MechanismConfig',
    'MechanismDiagnostics',
    'MultiDataGenerationResult',
    'MultiTableConfig',
    'MultiTableMechanism',
    'NumericalAttribute',
    'NumericalInitializer',
    'NumericalInitializerConfig',
    'NumericalMeasurement',
    'OpenSetCategoricalAttribute',
    'OpenSetInitializer',
    'OpenSetInitializerConfig',
    'OpenSetMeasurement',
    'PrivacyReport',
    'SWIFT',
    'SWIFTConfig',
    'Schema',
    'TabularCodec',
    'TabularConfig',
    'TabularMechanism',
    'TabularSynthesizer',
    'api',
    'calibrate',
    'checkpoint',
    'constraints',
    'create_initializers',
    'discrete_mechanisms',
    'domain',
    'encode_marginals',
    'from_yaml',
    'generate_synthetic_data_from_marginals',
    'infer_categorical_domain',
    'local_mode',
    'postprocessing',
    'relational',
    'reporting',
    'serialize',
    'to_yaml',
    'with_group_size',
]
