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

"""Implementations of mechanisms that operate over discrete data.

Note: This mechanism is not intended to be called directly. It should typically
be used within `DiscreteSynthesizer` or `TabularSynthesizer`. Users who call it
directly will miss out on features like 1-way measurement selection and domain
compression.
"""

# pylint: disable=g-importing-member

from dpsynth.core._mechanism import CalibratedMechanism
from dpsynth.core._mechanism import MechanismConfig
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

# Backwards-compatible aliases.
AIMMechanism = AIMConfig
DirectMechanism = DirectConfig
IndependentMechanism = IndependentConfig
MSTMechanism = MSTConfig
SWIFTMechanism = SWIFTConfig
