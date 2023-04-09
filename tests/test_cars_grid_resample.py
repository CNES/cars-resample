#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of cars-resample
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
#
"""Tests for `cars-resample` package."""

import numpy as np

# Third party imports
import pytest

import resample


@pytest.fixture
def synthetic_grid():
    """Generate synthetic grid."""
    cols = np.arange(4 + 2 * 2)
    rows = np.arange(5 + 2 * 2)
    grid = np.array(np.meshgrid(cols, rows)).astype(float)
    return grid


@pytest.fixture
def synthetic_image():
    """Generate synthetic image."""
    image = np.arange((5 + 2 * 2) * (4 + 2 * 2)).reshape(
        (1, 5 + 2 * 2, 4 + 2 * 2)
    )
    return image


def test_synthetic(
    synthetic_image, synthetic_grid
):  # pylint: disable=redefined-outer-name
    """Test on synthetic data."""
    out = resample.source_to_target(
        synthetic_image, synthetic_grid, oversampling=1
    )
    assert np.array_equal(out, synthetic_image)
