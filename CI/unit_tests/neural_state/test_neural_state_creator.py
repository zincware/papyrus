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
from numpy.testing import assert_raises

from papyrus.neural_state import NeuralStateCreator


class TestNeuralStateCreator:
    """
    Test the NeuralStateCreator class.
    """

    def test_init(self):
        """
        Test the constructor method of the NeuralStateCreator class.
        """

        def network_apply_fn(params: dict, data: dict):
            return np.arange(10)

        def ntk_apply_fn(params: dict, data: dict):
            return np.arange(10)

        neural_state_creator = NeuralStateCreator(
            network_apply_fn=network_apply_fn,
            ntk_apply_fn=ntk_apply_fn,
        )
        assert neural_state_creator.apply_fns == {
            "predictions": network_apply_fn,
            "ntk": ntk_apply_fn,
        }
        assert neural_state_creator.data_keys == ["inputs", "targets"]

    def test_apply(self):
        """
        Test the apply method of the NeuralStateCreator class.
        """

        def network_apply_fn(params: dict, data: dict):
            return np.arange(10)

        def ntk_apply_fn(params: dict, data: dict):
            return np.arange(10)

        neural_state_creator = NeuralStateCreator(
            network_apply_fn=network_apply_fn,
            ntk_apply_fn=ntk_apply_fn,
        )

        neural_state = neural_state_creator(
            params={},
            data={"inputs": np.arange(2), "targets": np.arange(2)},
            loss=np.arange(5),
        )
        assert np.all(neural_state.predictions == np.arange(10))
        assert np.all(neural_state.ntk == np.arange(10))
        assert np.all(neural_state.loss == np.arange(5))

        # Test with different data keys
        neural_state_creator = NeuralStateCreator(
            network_apply_fn=network_apply_fn,
            ntk_apply_fn=ntk_apply_fn,
            data_keys=["data", "labels"],
        )
        neural_state = neural_state_creator(
            params={},
            data={"data": np.arange(2), "labels": np.arange(2)},
            loss=np.arange(5),
        )
        assert np.all(neural_state.predictions == np.arange(10))
        assert np.all(neural_state.ntk == np.arange(10))
        assert np.all(neural_state.loss == np.arange(5))

        # Test with different data keys and missing data
        neural_state_creator = NeuralStateCreator(
            network_apply_fn=network_apply_fn,
            ntk_apply_fn=ntk_apply_fn,
            data_keys=["data", "labels"],
        )
        assert_raises(
            KeyError,
            neural_state_creator,
            params={},
            data={"inputs": np.arange(2), "targets": np.arange(2)},
            loss=np.arange(5),
        )
