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

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class NeuralState:
    """
    Data class to represent the state of a neural network.

    A neural network state can be represented in various ways. NeuralState offers a
    structured solution to represent the state of a neural network in terms of different
    properties.
    If the default properties are not sufficient, the user can extend this class to
    include more. In general, a property of a neural state can be any type of data, as
    long as it is formatted as `List[Any]` or `np.array[Any]`.

    Attributes
    ----------
    loss: Optional[List[np.ndarray]]
            The loss of a neural network.
    accuracy: Optional[List[np.ndarray]]
            The accuracy of a neural network.
    predictions: Optional[List[np.ndarray]]
            The predictions of a neural network.
    targets: Optional[List[np.ndarray]]
            The targets of a neural network.
    ntk: Optional[List[np.ndarray]]
            The neural tangent kernel of a neural network.
    """

    loss: Optional[List[np.ndarray]] = None
    accuracy: Optional[List[np.ndarray]] = None
    predictions: Optional[List[np.ndarray]] = None
    targets: Optional[List[np.ndarray]] = None
    ntk: Optional[List[np.ndarray]] = None

    def get_dict(self) -> dict:
        """
        Get a dictionary representation of the neural state.

        Only return the properties that are not None.

        Returns
        -------
        dict
                A dictionary representation of the neural state.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}
