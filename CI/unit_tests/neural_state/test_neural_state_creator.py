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

from papyrus.neural_state import NeuralStateCreator


class TestNeuralStateCreator:
    def test_init(self):
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

    def test_apply(self):
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
            data={},
            loss=np.arange(5),
        )
        assert np.all(neural_state.predictions == np.arange(10))
        assert np.all(neural_state.ntk == np.arange(10))
        assert np.all(neural_state.loss == np.arange(5))
