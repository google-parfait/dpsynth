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

# pylint: disable=g-importing-member,g-bad-import-order,g-import-not-at-top
__version__ = '0.5.0'

# Subpackages and legacy module aliases.
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth import relational

# Domain and schema specification.
from dpsynth.constraints import Constraint
from dpsynth.domain import AttributeType
from dpsynth.domain import CategoricalAttribute
from dpsynth.domain import FreeFormTextAttribute
from dpsynth.domain import NumericalAttribute
from dpsynth.domain import OpenSetCategoricalAttribute
from dpsynth.domain import Schema

# Core mechanism lifecycle and utilities.
from dpsynth._calibration import calibrate
from dpsynth._calibration import with_group_size
from dpsynth._checkpoint import checkpoint
from dpsynth.api import CalibratedMechanism
from dpsynth.api import MechanismConfig
from dpsynth.serialize import from_yaml
from dpsynth.serialize import to_yaml

# Privacy and auditing reports.
from dpsynth.reporting import AuditReport
from dpsynth.reporting import PrivacyReport
from dpsynth.reporting import TradeOffCurve

# Base discrete mechanisms.
from dpsynth.discrete_mechanisms.aim import AIMConfig
from dpsynth.discrete_mechanisms.direct import DirectConfig
from dpsynth.discrete_mechanisms.discrete import DiscreteConfig
from dpsynth.discrete_mechanisms.discrete import DiscreteMechanism
from dpsynth.discrete_mechanisms.independent import IndependentConfig
from dpsynth.discrete_mechanisms.mst import MSTConfig
from dpsynth.discrete_mechanisms.swift import SWIFTConfig

# Single-table tabular synthesis.
from dpsynth.data_generation_v3 import DataGenerationResult
from dpsynth.data_generation_v3 import TabularConfig
from dpsynth.data_generation_v3 import TabularMechanism

# Multi-table relational synthesis.
ForeignKeyRelation = relational.ForeignKeyRelation
MultiDataGenerationResult = relational.MultiDataGenerationResult
MultiTableConfig = relational.MultiTableConfig
MultiTableMechanism = relational.MultiTableMechanism

# Subpackages with optional heavyweight dependencies (such as dpsynth.training
# and dpsynth.adapters) are intentionally not imported here so they are loaded
# lazily only when explicitly imported by the caller.

# Route MBI callback logs (which use print-style formatting) to
# absl.logging.info.
from absl import logging
import mbi

if hasattr(mbi, 'callbacks') and hasattr(mbi.callbacks, 'set_log_fn'):
  mbi.callbacks.set_log_fn(
      lambda *args, sep=' ', **kwargs: logging.info(
          sep.join(str(a) for a in args)
      )
  )

__all__ = [
    'AIMConfig',
    'AttributeType',
    'AuditReport',
    'CalibratedMechanism',
    'CategoricalAttribute',
    'Constraint',
    'DataGenerationResult',
    'DirectConfig',
    'DiscreteConfig',
    'DiscreteMechanism',
    'ForeignKeyRelation',
    'FreeFormTextAttribute',
    'IndependentConfig',
    'MSTConfig',
    'MechanismConfig',
    'MultiDataGenerationResult',
    'MultiTableConfig',
    'MultiTableMechanism',
    'NumericalAttribute',
    'OpenSetCategoricalAttribute',
    'PrivacyReport',
    'SWIFTConfig',
    'Schema',
    'TabularConfig',
    'TabularMechanism',
    'TradeOffCurve',
    'calibrate',
    'checkpoint',
    'from_yaml',
    'relational',
    'to_yaml',
    'with_group_size',
]
