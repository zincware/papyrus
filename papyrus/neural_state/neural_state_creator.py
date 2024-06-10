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

from papyrus.neural_state.neural_state import NeuralState


class NeuralStateCreator:
    """
    Class creating a neural state.

    The NeuralStateCreator class serves as instance mapping data and parameter state to
    a NeuralState instance using a set of apply functions. The apply functions.
    These apply functions are e.g. the neural network forward pass or the neural tangent
    kernel computation.

    Attributes
    ----------
    apply_fns : dict
            A dictionary of apply functions that map the data and parameter state to a
            NeuralState instance.
    """

    def __init__(self, network_apply_fn: callable, ntk_apply_fn: callable):
        """
        Initialize the NeuralStateCreator instance.

        Parameters
        ----------
        network_apply_fn : callable
                The apply function that maps the data and parameter state to a
                NeuralState instance.
        ntk_apply_fn : callable
                The apply function that maps the data and parameter state to a
                NeuralState instance.
        """
        self.apply_fns = {
            "predictions": network_apply_fn,
            "ntk": ntk_apply_fn,
        }

    def __call__(self, params: dict, data: dict, **kwargs) -> NeuralState:
        """
        Call the NeuralStateCreator instance.

        Parameters
        ----------
        params : dict
                A dictionary of parameters that are used in the apply functions.
        data : dict
                A dictionary of data that is used in the apply functions.
        kwargs : Any
                Additional keyword arguments that are directly added to the
                neural state.

        Returns
        -------
        NeuralState
                The neural state that is created by the apply functions.
        """
        neural_state = NeuralState()

        for key, apply_fn in self.apply_fns.items():
            neural_state.__setattr__(key, apply_fn(params, data))

        for key, value in kwargs.items():
            neural_state.__setattr__(key, value)

        return neural_state
