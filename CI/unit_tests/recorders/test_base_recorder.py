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
"""

import os

import numpy as np
from numpy.testing import assert_array_equal

from papyrus.measurements import BaseMeasurement
from papyrus.recorders import BaseRecorder


class DummyMeasurement1(BaseMeasurement):
    """
    Dummy class to test the BaseRecorder class.
    """

    def __init__(self, name: str, rank: int):
        """
        Constructor method for the DummyMeasurement class.

        For more information see the BaseMeasurement class.
        """
        super().__init__(name, rank)

    def apply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Apply method for the DummyMeasurement class.

        For more information see the BaseMeasurement class.
        """
        return a + b


class DummyMeasurement2(BaseMeasurement):
    """
    Dummy class to test the BaseRecorder class.
    """

    def __init__(self, name: str, rank: int):
        """
        Constructor method for the DummyMeasurement class.

        For more information see the BaseMeasurement class.
        """
        super().__init__(name, rank)

    def apply(self, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Apply method for the DummyMeasurement class.

        For more information see the BaseMeasurement class.
        """
        return np.sum(b + c, axis=-1)


class TestBaseRecorder:
    """
    Test the base recorder class.
    """

    @classmethod
    def setup_class(cls):
        """
        Setup the test class.
        """
        cls.measurement_1 = DummyMeasurement1("dummy_1", 1)
        cls.measurement_2 = DummyMeasurement2("dummy_2", 2)

        # Each of keys a, b, c has shape (3, 10, 5), which could e.g. represent
        # Precitions of 3 Sub-scamples of 10 data points with 5 features.
        cls.neural_state = {
            "a": np.ones(shape=(3, 10, 5)),
            "b": np.zeros(shape=(3, 10, 5)),
            "c": 2 * np.ones(shape=(3, 10, 5)),
        }

    def test_init(self):
        """
        Test the constructor method of the BaseRecorder class.
        """
        # Test the constructor method
        name = "test"
        storage_path = "test_path"
        recorder = BaseRecorder(
            name, storage_path, [self.measurement_1, self.measurement_2], 10
        )
        assert recorder.name == name
        assert recorder.storage_path == storage_path
        assert recorder.measurements == [self.measurement_1, self.measurement_2]
        assert recorder.chunk_size == 10

    def test_neural_state_keys(self):
        """
        Test the neural_state_keys attribute of the BaseRecorder class.
        """
        # Test whether the neural_state_keys attribute is correctly set
        name = "test"
        storage_path = "test_path"
        recorder = BaseRecorder(
            name, storage_path, [self.measurement_1, self.measurement_2], 10
        )
        assert set(recorder.neural_state_keys) == set(["a", "b", "c"])

    def test_measure(self):
        """
        Test the measure method of the BaseRecorder class.
        """

        # Test measuring for the first time

        name = "test"
        storage_path = "test_path"
        recorder = BaseRecorder(
            name, storage_path, [self.measurement_1, self.measurement_2], 10
        )
        recorder._measure(**self.neural_state)

        assert recorder._results.keys() == {"dummy_1", "dummy_2"}

        assert_array_equal(recorder._results["dummy_1"], np.ones(shape=(1, 3, 10, 5)))
        assert_array_equal(recorder._results["dummy_2"], 10 * np.ones(shape=(1, 3, 10)))

        # Test measuring for the second time
        recorder._measure(**self.neural_state)
        assert_array_equal(
            recorder._results["dummy_1"], 1 * np.ones(shape=(2, 3, 10, 5))
        )
        assert_array_equal(recorder._results["dummy_2"], 10 * np.ones(shape=(2, 3, 10)))

    def test_write_read(self):
        """
        Test the write and read methods of the BaseRecorder class.
        """
        # Create a temporary directory
        os.makedirs("temp/", exist_ok=True)
        name = "test"
        storage_path = "temp/"
        recorder = BaseRecorder(
            name, storage_path, [self.measurement_1, self.measurement_2], 10
        )

        # Test writing and reading
        recorder._measure(**self.neural_state)
        recorder._write(recorder._results)
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(1, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(1, 3, 10)))

        # Delete temporary directory
        os.system("rm -r temp/")

    def test_store(self):
        """
        Test the store method of the BaseRecorder class.
        """
        # Create a temporary directory
        os.makedirs("temp/", exist_ok=True)
        name = "test"
        storage_path = "temp/"
        recorder = BaseRecorder(
            name, storage_path, [self.measurement_1, self.measurement_2], 10
        )

        # Test storing
        recorder._measure(**self.neural_state)
        recorder._store(0)
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(1, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(1, 3, 10)))

        # Test storing again
        recorder._measure(**self.neural_state)
        recorder._store(10)
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        print(data["dummy_1"].shape)
        assert_array_equal(data["dummy_1"], np.ones(shape=(2, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(2, 3, 10)))

        # Delete temporary directory
        os.system("rm -r temp/")