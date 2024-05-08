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

import numpy as np

from papyrus.measurements.base_measurement import BaseMeasurement


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
