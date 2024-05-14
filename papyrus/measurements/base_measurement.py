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

from abc import ABC
from inspect import signature
from typing import List

import numpy as np


class BaseMeasurement(ABC):
    """
    Base class for all measurements.

    A measurement is a class that records an aspect of the learning process.

    Note
    ----
    All measurements should inherit from this class and implement the apply method.

    Measurements that return arrays with sizes that depend on the number of inputs
    **cannot** be applied on varying number of inputs. This is because the number of
    dimensions of the input need to be same for all subsequent calls, otherwise an error
    will be raised when storing the results in the database.

    Attributes
    ----------
    name : str
            The name of the measurement, defining how the instance in the database will
            be identified.
    rank : int
            The rank of the measurement, defining the tensor order of the measurement.
    public : bool
            Boolean flag to indicate whether the measurement resutls will be accessible
            via a public attribute of the recorder.
    neural_state_keys : List[str]
            The keys of the neural state that the measurement takes as input.
            A neural state is a dictionary of numpy arrays that represent the state of
            a neural network. The keys are the names of the kwargs of the apply method.
    """

    def __init__(self, name: str, rank: int, public: bool = False):
        """
        Constructor method of the BaseMeasurement class.

        name : str
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        """
        self.name = name
        # self.takes_states = takes_states
        self.rank = rank
        self.public = public

        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("Rank must be a positive integer.")

        # Get the neural state keys that the measurement takes as input
        self.neural_state_keys: List[str] = []
        self.neural_state_keys = self._get_apply_signature()

    def _get_apply_signature(self):
        """
        Method to get the signature of the apply method.

        This method is used to get the signature of the apply method, in order to
        determine the keys of the neural state that the measurement takes as input.

        Returns
        -------
        List[str]
                The keys of the neural state that the measurement takes as input.
        """
        return list(signature(self.apply).parameters.keys())

    def apply(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Method to perform a measurement.

        This method should be implemented in the child class.
        It can take any number of arguments and keyword arguments and should return
        the result of the function applied to the measurement.

        Note that this method performs a single measurement, on the diven input.

        Parameters
        ----------
        *args : np.ndarray
                The arguments to the function.
        **kwargs : np.ndarray
                The keyword arguments to the function.


        Returns
        -------
        np.ndarray
            The result of the function applied to the measurement.
        """
        raise NotImplementedError("Implemented in child class.")

    def __call__(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Method to perform the measurement.

        This method calls the apply_fn method on each input and returns the result.

        It can take any number of arguments and keyword arguments and should return
        the measurement result.
        All inputs should be numpy arrays, with their first dimension indicating the
        number of eqivalent measurements to be performed. The remaining dimensions
        should be the same for all inputs, as they are the arguments to the function.
        In case of one input, the first dimension should be 1.

        Returns
        -------
        np.ndarray
                The result of the measurement.
        """
        # Get the number of arguments
        num_args = len(args)
        # Get the keys and values of the keyword arguments if any
        keys = list(kwargs.keys())
        vals = list(kwargs.values())
        # Zip the arguments and values
        z = zip(*args, *vals)

        # Perform the measurement on each set of inputs
        return np.array(
            [self.apply(*i[:num_args], **dict(zip(keys, i[num_args:]))) for i in z]
        )
