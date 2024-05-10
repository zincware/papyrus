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
from numpy.testing import assert_almost_equal, assert_array_equal, assert_raises

from papyrus.utils import (
    compute_shannon_entropy,
    compute_trace,
    compute_von_neumann_entropy,
)


class TestMatrixUtils:
    """
    Test suite for the matrix utils.
    """

    def test_compute_trace(self):
        """
        Test the computation of the trace.
        """
        vector = np.random.rand(10)
        matrix = np.diag(vector)

        # Test the trace without normalization
        trace = compute_trace(matrix, normalize=False)
        assert trace == np.sum(vector)

        # Test the trace with normalization
        trace = compute_trace(matrix, normalize=True)
        assert trace == np.sum(vector) / 10

    def test_shannon_entropy(self):
        """
        Test the Shannon entropy.
        """
        dist = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert_almost_equal(compute_shannon_entropy(dist), np.log(5))
        assert_almost_equal(compute_shannon_entropy(dist, normalize=True), 1.0)

        dist = np.array([0, 0, 0, 0, 1])
        assert compute_shannon_entropy(dist) == 0
        assert compute_shannon_entropy(dist, normalize=True) == 0

        dist = np.array([0, 0, 0, 0.5, 0.5])
        assert compute_shannon_entropy(dist) == np.log(2)
        s = compute_shannon_entropy(dist, normalize=True)
        assert s == np.log(2) / np.log(5)

    def test_compute_von_neumann_entropy(self):
        """
        Test the computation of the von-Neumann entropy.
        """
        matrix = np.eye(2) * 0.5
        entropy = compute_von_neumann_entropy(matrix=matrix, effective=False)
        assert entropy == np.log(2)

        entropy = compute_von_neumann_entropy(matrix=matrix, effective=True)
        assert entropy == 1
