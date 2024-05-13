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

from papyrus.measurements import BaseMeasurement


class DummyMeasurement(BaseMeasurement):
    """
    Dummy measurement class for testing.
    """

    def __init__(self, name: str, rank: int, public: bool = False):
        super().__init__(name, rank, public)

    def apply(
        self, a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if c is not None:
            return a + b + c
        return a + b


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

    def test_call(self):
        """
        Test the call method of the BaseMeasurement class.
        """
        # Test the call method
        name = "test"
        rank = 1
        public = False
        measurement = BaseMeasurement(name, rank, public)

        # Test the apply method
        with pytest.raises(NotImplementedError):
            measurement.apply()

        # Set an exmaple apply method
        def apply(
            a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray] = None
        ) -> np.ndarray:
            if c is not None:
                return a + b + c
            return a + b

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        c = np.array([[9, 10], [11, 12]])

        # Test the call method with only arguments
        measurement = DummyMeasurement(name, rank, public)
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
        assert np.allclose(result, a + b)

    def test_apply(self):
        """
        Test the apply method of the BaseMeasurement class.
        """
        # Create an example apply method and initialize the measurement

        def apply(
            a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray] = None
        ) -> np.ndarray:
            if c is not None:
                return a + b + c
            return a + b

        name = "test"
        rank = 1
        public = False
        measurement = DummyMeasurement(name, rank, public)
        assert measurement.neural_state_keys == ["a", "b", "c"]
