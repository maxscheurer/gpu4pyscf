# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

"""
pytest configuration for int3c1e benchmarks.

This module provides fixtures for pytest-harvest result collection.
"""

import pytest

# Try to import pytest-harvest, but make it optional
try:
    from pytest_harvest import saved_fixture
    HARVEST_AVAILABLE = True
except ImportError:
    HARVEST_AVAILABLE = False


if HARVEST_AVAILABLE:
    @pytest.fixture
    @saved_fixture
    def results_bag():
        """
        Fixture to collect benchmark results per test.

        Results stored in this dictionary are collected by pytest-harvest
        into the fixture_store.

        Returns
        -------
        dict
            Empty dictionary to store benchmark results
        """
        return {}

else:
    # Fallback when pytest-harvest is not available
    @pytest.fixture
    def results_bag():
        """Dummy fixture when pytest-harvest is not available."""
        return {}
