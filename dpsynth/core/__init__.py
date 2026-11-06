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

"""Foundational abstractions for DPSynth."""

from . import _calibration
from . import _checkpoint
from . import _mechanism
from . import constraints
from . import domain
from . import reporting
from . import serialize

# Domain and schema specification.
AttributeType = domain.AttributeType
CategoricalAttribute = domain.CategoricalAttribute
Constraint = constraints.Constraint
FreeFormTextAttribute = domain.FreeFormTextAttribute
NumericalAttribute = domain.NumericalAttribute
OpenSetCategoricalAttribute = domain.OpenSetCategoricalAttribute
Schema = domain.Schema

# Core mechanism lifecycle and utilities.
CalibratedMechanism = _mechanism.CalibratedMechanism
MechanismConfig = _mechanism.MechanismConfig
calibrate = _calibration.calibrate
checkpoint = _checkpoint.checkpoint
from_yaml = serialize.from_yaml
to_yaml = serialize.to_yaml
with_group_size = _calibration.with_group_size

# Privacy and auditing reports.
AuditReport = reporting.AuditReport
PrivacyReport = reporting.PrivacyReport
TradeOffCurve = reporting.TradeOffCurve

__all__ = [
    "AttributeType",
    "AuditReport",
    "CalibratedMechanism",
    "CategoricalAttribute",
    "Constraint",
    "FreeFormTextAttribute",
    "MechanismConfig",
    "NumericalAttribute",
    "OpenSetCategoricalAttribute",
    "PrivacyReport",
    "Schema",
    "TradeOffCurve",
    "calibrate",
    "checkpoint",
    "from_yaml",
    "to_yaml",
    "with_group_size",
]
