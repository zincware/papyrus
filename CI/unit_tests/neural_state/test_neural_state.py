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

from papyrus.neural_state import NeuralState


class TestNeuralState:

    def test_init(self):

        neural_state = NeuralState()
        assert neural_state.loss is None
        assert neural_state.accuracy is None
        assert neural_state.predictions is None
        assert neural_state.targets is None
        assert neural_state.ntk is None

        neural_state = NeuralState(
            loss=[],
            accuracy=[],
            predictions=[],
            targets=[],
            ntk=[],
        )
        assert neural_state.loss == []
        assert neural_state.accuracy == []
        assert neural_state.predictions == []
        assert neural_state.targets == []
        assert neural_state.ntk == []

    def test_get_dict(self):

        neural_state = NeuralState()
        assert neural_state.get_dict() == {}

        neural_state = NeuralState(
            loss=[],
            accuracy=[],
            predictions=[],
            targets=[],
            ntk=[],
        )
        assert neural_state.get_dict() == {
            "loss": [],
            "accuracy": [],
            "predictions": [],
            "targets": [],
            "ntk": [],
        }
