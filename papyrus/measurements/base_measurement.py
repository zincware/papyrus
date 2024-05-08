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

import numpy as np


class BaseMeasurement(ABC):
    """
    Base class for all measurements.

    A measurement is a class that records an aspect of the learning process.
    """

    def __init__(self, name: str, rank: int):
        """
        Constructor method of the BaseMeasurement class.

        name : str
            The name of the measurement, defining how the instance in the database will
            be identified.
        rank : int
            The rank of the measurement, defining the order in which the measurement
            will be recorded.
        """
        self.name = name
        self.rank = rank

        if not isinstance(self.rank, int) and self.rank < 1:
            raise ValueError("Rank must be a positive integer.")

    def apply_fn(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
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

        This method should be implemented in the child class.
        It can take any number of arguments and keyword arguments and should return
        the measurement result.

        All inputs should be numpy arrays, with their first dimension indicating the
        number of eqivalent measurements to be performed.

        Returns
        -------
        np.ndarray
            The result of the measurement.
        """
        # Get the keys and values of the keyword arguments
        keys, vals = zip(*kwargs.items())
        # Zip the arguments and values
        z = zip(*args, *vals)
        # Get the length of the keys to split the zipped values later
        l = len(keys)
        # Perform the measurement on each set of inputs
        return np.array([self.apply_fn(i[:l], dict(zip(keys, i[l:]))) for i in z])
