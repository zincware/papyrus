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

from typing import Dict, List

import numpy as np

from papyrus.measurements.base_measurement import BaseMeasurement
from papyrus.neural_state.neural_state import NeuralState
from papyrus.neural_state.neural_state_creator import NeuralStateCreator as NSC
from papyrus.recorders.base_recorder import BaseRecorder


class DatasetRecorder(BaseRecorder):
    """
    Recorder for a fixed dataset.
    """

    def __init__(
        self,
        name: str,
        storage_path: str,
        measurements: List[BaseMeasurement],
        chunk_size: int,
        neural_state_creator: NSC,
        dataset: Dict[str, np.ndarray],
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
        neural_state_creator : NSC
                The neural state creator that will be used to create the neural state.
                It includes all apply methods that map data and parameters to a neural
                state.
        dataset : Dict[str, np.ndarray]
                The dataset that will be used to create the neural state.
                It needs to be a dictionary of numpy arrays with the following keys:
                - "inputs": The inputs of the dataset.
                - "targets": The targets of the dataset.
        """
        super().__init__(name, storage_path, measurements, chunk_size)
        self.neural_state_creator = neural_state_creator
        self.dataset = dataset

    def record(self, epoch: int, params: Dict[str, np.ndarray], **kwargs):
        """
        Perform the recording of a neural state.

        Recording is done by measuring and storing the measurements to a database.

        Parameters
        ----------
        epoch : int
                The epoch of recording.
        params : Dict[str, np.ndarray]
                The parameters of the neural network.
        kwargs : Any
                Additional keyword arguments that are directly added to the neural
                state.

        Returns
        -------
        result : np.ndarray
                The result of the recorder.
        """
        neural_state: NeuralState = self.neural_state_creator(
            params, self.dataset, kwargs
        )
        self._measure(**neural_state.get_dict())
        self._store(epoch)
