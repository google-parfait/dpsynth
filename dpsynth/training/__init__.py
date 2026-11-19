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

"""Public API for differentially private model training and fine-tuning.

Gemma-specific helpers (model loading, sampling, and ``DPFineTunerConfig``)
live in the optional ``dpsynth.training.gemma`` submodule, which requires Tunix.
"""

# pylint: disable=g-importing-member

from dpsynth.training._sequence_model import LossNormalization
from dpsynth.training._sequence_model import SequenceModel
from dpsynth.training._sft import DPFineTuner
from dpsynth.training._sft import FineTuneResult
from dpsynth.training._tabular_transformer import TabularTransformer
from dpsynth.training._tabular_transformer import TabularTransformerConfig
from dpsynth.training._trainer import DPTrainer
from dpsynth.training._trainer import DPTrainerConfig
from dpsynth.training._trainer import TrainingResult

__all__ = [
    'DPFineTuner',
    'DPTrainer',
    'DPTrainerConfig',
    'FineTuneResult',
    'LossNormalization',
    'SequenceModel',
    'TabularTransformer',
    'TabularTransformerConfig',
    'TrainingResult',
]
