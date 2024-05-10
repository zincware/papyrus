"""
papyrus: a lightweight Python library to record neural learning.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Summary
-------
Test the base measurement class.
"""

from typing import Optional

import numpy as np
import pytest

from papyrus.measurements.base_measurement import BaseMeasurement


class TestBaseMeasurement:
    """
    Test the base measurement class.
    """

    def test_init(self):
        """
        Test the constructor method of the BaseMeasurement class.
        """
        # Test the constructor method
        name = "test"
        rank = 1
        public = False
        measurement = BaseMeasurement(name, rank, public)
        assert measurement.name == name
        assert measurement.rank == rank
        assert measurement.public == public

        # Test the rank parameter
        with pytest.raises(ValueError):
            BaseMeasurement(name, -1, public)
        with pytest.raises(ValueError):
            BaseMeasurement(name, 0, public)

    def test_call(self):
        """
        Test the call method of the BaseMeasurement class.
        """
        # Test the call method
        name = "test"
        rank = 1
        public = False
        measurement = BaseMeasurement(name, rank, public)

        # Test the apply_fn method
        with pytest.raises(NotImplementedError):
            measurement.apply_fn()

        # Set an exmaple apply_fn method
        def apply_fn(
            a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray] = None
        ) -> np.ndarray:
            if c is not None:
                return a + b + c
            return a + b

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        c = np.array([[9, 10], [11, 12]])

        # Test the call method with only arguments
        measurement.apply_fn = apply_fn
        result = measurement(a, b)
        assert np.allclose(result, a + b)
        result = measurement(a, b, c)
        assert np.allclose(result, a + b + c)

        # Test the call method with only keyword arguments
        result = measurement(a=a, b=b)
        assert np.allclose(result, a + b)
        result = measurement(a=a, b=b, c=c)
        assert np.allclose(result, a + b + c)

        # Test the call method with both arguments and keyword arguments
        result = measurement(a, b, c=c)
        assert np.allclose(result, a + b + c)
        result = measurement(a, b=b)
        print(result)
        assert np.allclose(result, a + b -1)