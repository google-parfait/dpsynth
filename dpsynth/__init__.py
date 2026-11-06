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
from dpsynth import core
from dpsynth import discrete_mechanisms
from dpsynth import relational

domain = core.domain

# Domain and schema specification.
AttributeType = core.AttributeType
CategoricalAttribute = core.CategoricalAttribute
Constraint = core.Constraint
FreeFormTextAttribute = core.FreeFormTextAttribute
NumericalAttribute = core.NumericalAttribute
OpenSetCategoricalAttribute = core.OpenSetCategoricalAttribute
Schema = core.Schema

# Core mechanism lifecycle and utilities.
CalibratedMechanism = core.CalibratedMechanism
MechanismConfig = core.MechanismConfig
calibrate = core.calibrate
checkpoint = core.checkpoint
from_yaml = core.from_yaml
to_yaml = core.to_yaml
with_group_size = core.with_group_size

# Privacy and auditing reports.
AuditReport = core.AuditReport
PrivacyReport = core.PrivacyReport
TradeOffCurve = core.TradeOffCurve

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
from dpsynth.statistical.postprocessing import generate_synthetic_data_from_marginals

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
    'generate_synthetic_data_from_marginals',
    'relational',
    'to_yaml',
    'with_group_size',
]
