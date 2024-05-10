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
Module containing default measurements for recording neural learning.
"""

import numpy as np

from papyrus.measurements.base_measurement import BaseMeasurement


class NTKTrace(BaseMeasurement):
    """
    Measurement class to record the trace of the NTK.
    """

    def __init__(
        self,
        name: str = "ntk_trace",
        rank: int = 1,
        public: bool = False,
        normalise: bool = True,
    ):
        """
        Constructor method of the NTKTrace class.

        Parameters
        ----------
        name : str (default="ntk_trace")
            The name of the measurement, defining how the instance in the database will
            be identified.
        rank : int (default=1)
            The rank of the measurement, defining the tensor order of the measurement.
        normalise : bool (default=True)
            Boolean flag to indicate whether the trace of the NTK will be normalised by
            the size of the NTK matrix.
        """
        super().__init__(name, rank, public)
        self.normalise = normalise

    def apply_fn(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to compute the trace of the NTK.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.

        Returns
        -------
        np.ndarray
            The trace of the NTK
        """
        pass


class NTKCorrelationEntropy(BaseMeasurement):
    """
    Measurement class to record the correlation entropy of the NTK.

    Parameters
    ----------
    rank : int
        The rank of the measurement, defining the order in which the measurement
        will be recorded.
    """

    def __init__(self, name: str = "ntk_cross_entropy"):
        """
        Constructor method of the NTKCrossEntropy class.

        rank : int
            The rank of the measurement, defining the order in which the measurement
            will be recorded.
        """
        rank = 1
        super().__init__(name, rank=rank)

    def apply_fn(self, ntk: np.ndarray) -> np.ndarray:
        """
        TODO: Implement this method.
        """
        pass
