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
Base module for a recorder.
"""

from abc import ABC
from typing import List

import numpy as np

from papyrus.measurements.base_measurement import BaseMeasurement
from papyrus.recorders.data_storage import DataStorage


class BaseRecorder(ABC):
    """
    Base class for a recorder.

    A recorder is a class that applies and stores measurements. It is used to record
    the learning process of a neural network.

    Recorders that return arrays with sizes that depend on the number of inputs
    **cannot** be applied on varying number of inputs. This is because the number of
    dimensions of the input need to be same for all subsequent calls, otherwise an error
    will be raised when storing the results in the database.

    Attributes
    ----------
    name : str
            The name of the recorder, defining the name of the file the data will be
            stored in.
    storage_path : str
            The path to the storage location of the recorder.
    measurements : List[BaseMeasurement]
            The measurements that the recorder will apply.
    chunk_size : int
            The size of the chunks in which the data will be stored.
    overwrite : bool (default=False)
            Whether to overwrite the existing data in the database.
    neural_state_keys : List[str]
            The keys of the neural state that the recorder takes as input.
            A neural state is a dictionary of numpy arrays that represent the state of
            a neural network.
    """

    def __init__(
        self,
        name: str,
        storage_path: str,
        measurements: List[BaseMeasurement],
        chunk_size: int,
        overwrite: bool = False,
    ):
        """
        Constructor method of the BaseRecorder class.

        Parameters
        ----------
        name : str
                The name of the recorder, defining the name of the file the data will be
                stored in.
        storage_path : str
                The path to the storage location of the recorder.
        measurements : List[BaseMeasurement]
                The measurements that the recorder will apply.
        chunk_size : int
                The size of the chunks in which the data will be stored.
        overwrite : bool (default=False)
                Whether to overwrite the existing data in the database.
        """
        self.name = name
        self.storage_path = storage_path
        self.measurements = measurements
        self.chunk_size = chunk_size
        self.overwrite = overwrite

        # Initialize the data storage
        self._data_storage = DataStorage(f"{self.storage_path}/{self.name}")

        # Read in neural state keys from measurements
        self.neural_state_keys = self._read_neural_state_keys()

        # Temporary storage for results
        self._init_internals()

        # Check for existing data and overwrite if necessary
        if self.overwrite:
            try:
                keys = self._data_storage.read_keys()
                for key in keys:
                    self._data_storage.del_dataset(key)
            except FileNotFoundError:
                pass

    def _read_neural_state_keys(self):
        """
        Read the neural state keys from the measurements.

        Updates the neural_state_keys attribute of the recorder with the keys of the
        neural state that the measurements take as input.
        """
        neural_state_keys = []
        for measurement in self.measurements:
            neural_state_keys.extend(measurement.neural_state_keys)
        return list(set(neural_state_keys))

    def _init_internals(self):
        """
        Initialize the temporary storage for the results.
        """
        self._results = {measurement.name: [] for measurement in self.measurements}
        # Reset the counter
        self._counter = 0

    def load(self):
        """
        Load the data from the database using np.load.

        Returns
        -------
        data : dict
                The data loaded from the database.
        """
        return self._data_storage.load(self._results.keys())

    def _measure(self, **neural_state):
        """
        Perform the measurement of a neural state.

        Parameters
        ----------
        **neural_state
                The neural state that the recorder will record.

        Returns
        -------
        Updates self._results containing the results of the measurements.
        """
        for measurement in self.measurements:
            # Get the neural state keys that the measurement takes as input
            sub_state = {
                key: neural_state[key] for key in measurement.neural_state_keys
            }
            # Apply the measurement
            result = measurement(**sub_state)
            # Store the result in the temporary storage
            self._results[measurement.name].append(result)

        # Increment the counter
        self._counter += 1

    def store(self, ignore_chunk_size=True):
        """
        Store the results of the measurements in the database.

        This method loads and writes the data to the database in chunks.
        """
        if self._counter % self.chunk_size == 0 or ignore_chunk_size:
            # Write the data back to the database
            self._data_storage.write(self._results)
            # Reinitialize the temporary storage
            self._init_internals()

    def gather(self):
        """
        Gather the results from the temporary storage and the database.

        Returns
        -------
        data : dict
                The gathered data.
        """
        # Load the data from the database
        try:
            loaded_data = self.load()
            # If the counter is 0, the temporary storage is empty
            if self._counter == 0:
                data = loaded_data
            # Check if loaded data is empty
            elif all([len(v) == 0 for v in loaded_data.values()]):
                data = self._results
            # Append the new data to the loaded data
            else:
                for key in self._results.keys():
                    loaded_data[key] = np.append(
                        loaded_data[key], self._results[key], axis=0
                    )
                data = loaded_data
        # If the file does not exist, create a new one
        except FileNotFoundError:
            data = self._results

        return data

    def record(self, neural_state: dict):
        """
        Perform the recording of a neural state.

        Recording is done by measuring and storing the measurements to a database.

        Parameters
        ----------
        epoch : int
                The epoch of recording.
        **neural_state
                The neural state that the recorder will record.

        Returns
        -------
        result : np.ndarray
                The result of the recorder.
        """
        self._measure(**neural_state)
        self.store(ignore_chunk_size=False)
