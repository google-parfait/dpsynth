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

"""Test the TensorFlow is loaded only after TFRecord-specific paths."""

import subprocess
import sys

from absl.testing import absltest

class LazyTeorflowImportTest(absltest.TestCase):
    
    def assert_imports_do_not_load(self, modules, forbidden_packages):
        script = (
            'import importlib\n'
            'import sys\n'
            f'modules = {modules!r}\n'
            f'forbidden_packages = {forbidden_packages!r}\n'
            '\n'
            'for module in modules:\n'
            '   importlib.import_module(module)\n'
            '   loaded = [\n'
            '       package\n'
            '       for package in forbidden_packages\n'
            '       if package in sys.modules\n'
            '   ]\n'
            '   if loaded:\n'
            '       raise AssertionError(\n'
            '           f"{module} eagerly imported: {loaded}"\n'
            '   )\n'
        )
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            check=False,
            text=True,
        )
        
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                'import subprocess failed.\n'
                f'stdout:\n{result.stdout}\n'
                f'stderr:\n{result.stderr}'
            ),
        )
    
    def test_pipeline_import_do_not_load_tensorflow(self):
        self.assert_imports_do_not_load(
            modules=(
                'dpsynth.data_generation',
                'dpsynth.pipeline_transformations.input_output',
            ),
            forbidden_packages=('tensorflow',),
        )

if __name__ == '__main__':
    absltest.main()