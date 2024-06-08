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

import tempfile

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

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
        storage_path = "."
        recorder = BaseRecorder(
            name, storage_path, [self.measurement_1, self.measurement_2], 10
        )
        assert recorder.name == name
        assert recorder.storage_path == storage_path
        assert recorder.measurements == [self.measurement_1, self.measurement_2]
        assert recorder.chunk_size == 10
        assert recorder.overwrite is False
        assert recorder._data_storage.database_path == f"{storage_path}{name}.h5"

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
        storage_path = "."
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

    def test_store(self):
        """
        Test the store method of the BaseRecorder class.
        """
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            4,
        )

        # Test storing
        recorder._measure(**self.neural_state)
        recorder.store()
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(1, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(1, 3, 10)))

        # Test storing again
        recorder._measure(**self.neural_state)
        recorder.store()
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(2, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(2, 3, 10)))
        # _results should be empty after storing
        assert recorder._results == {"dummy_1": [], "dummy_2": []}

        # Test storing with ignore_chunk_size=False
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)
        recorder.store(ignore_chunk_size=False)
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(2, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(2, 3, 10)))

        # Delete temporary directory
        temp_dir.cleanup()

    def test_counter(self):
        """
        Test the counter attribute of the BaseRecorder class.
        """
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            3,
        )

        # Test counter
        assert recorder._counter == 0
        recorder._measure(**self.neural_state)
        assert recorder._counter == 1
        recorder._measure(**self.neural_state)
        assert recorder._counter == 2
        recorder.store(
            ignore_chunk_size=False
        )  # It should not story due to the chunk size
        assert recorder._counter == 2
        recorder._measure(**self.neural_state)
        assert recorder._counter == 3
        recorder.store(ignore_chunk_size=False)  # It should store now
        assert recorder._counter == 0

        # Delete temporary directory
        temp_dir.cleanup()

    def test_gather(self):
        """
        Test the gather method of the BaseRecorder class.
        """
        # Test gather if stored and unstored data is present
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            3,
        )
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)
        recorder.store()
        recorder._measure(**self.neural_state)

        data = recorder.gather()
        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(4, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(4, 3, 10)))
        # Clear the temporary directory
        temp_dir.cleanup()

        # Test gather if only stored data is present
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            3,
        )
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)
        recorder.store()

        data = recorder.gather()
        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(3, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(3, 3, 10)))
        # Clear the temporary directory
        temp_dir.cleanup()

        # Test gather if only unstored data is present
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            3,
        )
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)
        recorder._measure(**self.neural_state)

        data = recorder.gather()
        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(3, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(3, 3, 10)))
        # Clear the temporary directory
        temp_dir.cleanup()

    def test_overwrite(self):
        """
        Test the overwrite attribute of the BaseRecorder class.
        """
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            3,
            overwrite=True,
        )

        # Measure and save data
        recorder._measure(**self.neural_state)
        recorder._data_storage.write(recorder._results)
        data = recorder.load()
        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(1, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(1, 3, 10)))

        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            2,
            overwrite=True,
        )
        assert_raises(KeyError, recorder.load)

        # Measure and save data again
        recorder._measure(**self.neural_state)
        # Should not store the data because the chunk size is 2
        recorder.store(ignore_chunk_size=False)
        recorder._measure(**self.neural_state)
        # Should store the data now
        recorder.store(ignore_chunk_size=False)
        data = recorder.load()

        assert set(data.keys()) == {"dummy_1", "dummy_2"}
        assert_array_equal(data["dummy_1"], np.ones(shape=(2, 3, 10, 5)))
        assert_array_equal(data["dummy_2"], 10 * np.ones(shape=(2, 3, 10)))

    def test_recoding_order(self):
        """
        Test the order of the recordings.
        """
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        name = "test"
        storage_path = temp_dir.name
        recorder = BaseRecorder(
            name,
            storage_path,
            [self.measurement_1, self.measurement_2],
            3,
        )

        # Prepare distinct neural states
        neural_state_1 = {
            "a": np.ones(shape=(3, 10, 5)),
            "b": np.ones(shape=(3, 10, 5)),
            "c": np.ones(shape=(3, 10, 5)),
        }
        neural_state_2 = {k: v * 2 for k, v in neural_state_1.items()}
        neural_state_3 = {k: v * 3 for k, v in neural_state_1.items()}

        # Measure and store data
        recorder._measure(**neural_state_1)
        recorder.store()
        recorder._measure(**neural_state_2)
        recorder.store()
        recorder._measure(**neural_state_3)
        recorder.store()

        # Gather data
        data = recorder.gather()
        print(data)

        # Check the order of the recordings
        assert_array_equal(data["dummy_1"][0], 2 * np.ones(shape=(3, 10, 5)))
        assert_array_equal(data["dummy_1"][1], 4 * np.ones(shape=(3, 10, 5)))
        assert_array_equal(data["dummy_1"][2], 6 * np.ones(shape=(3, 10, 5)))

        # Clear the temporary directory
        temp_dir.cleanup()
