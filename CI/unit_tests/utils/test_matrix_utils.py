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
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

from papyrus.utils.matrix_utils import compute_hermitian_eigensystem


class TestMatrixUtils:
    """
    Test the matrix utils class.
    """

    def test_compute_hermitian_eigensystem(self):
        """
        Test the computation of the eigensystem of a hermitian matrix.
        """
        # Create a dummy hermitian matrix
        # eigenvalues
        l1 = 1
        l2 = 2
        l3 = 3
        # eigenvectors
        v1 = np.array([1, 1, 0]) / np.sqrt(2)
        v2 = np.array([1, -1, 0]) / np.sqrt(2)
        v3 = np.array([0, 0, 1])
        # create the matrix
        M = l1 * np.outer(v1, v1.conj())
        M += l2 * np.outer(v2, v2.conj())
        M += l3 * np.outer(v3, v3.conj())

        # check if it is hermitian
        assert np.allclose(M, M.T.conj())

        # assert that the eigenvalues and eigenvectors are correct

        # normalize = False
        vals, vects = compute_hermitian_eigensystem(M, normalize=False)
        assert_array_almost_equal(vals, [3, 2, 1])
        assert_array_almost_equal(vects, np.array([v3, v2, v1]).T)

        # normalize = True
        vals, vects = compute_hermitian_eigensystem(M, normalize=True)
        assert_array_almost_equal(vals, [3 / 6, 2 / 6, 1 / 6])
        assert_array_almost_equal(vects, np.array([v3, v2, v1]).T)

    # def test_normalizing_covariance_matrix(self):
    #     """
    #     Test that the covariance matrix is correctly normalized.

    #     Returns
    #     -------
    #     We fix the diagonals and test whether it performs the correct operation. You
    #     should note that this is not a correctly normalized covariance matrix rather
    #     one that can be tested well to properly scale under the normalization procedure.
    #     """
    #     # 4x4 covariance matrix
    #     covariance_matrix = onp.random.uniform(low=0, high=3, size=(4, 4))

    #     # Fix diagonals
    #     for i in range(4):
    #         covariance_matrix[i, i] = i + 3

    #     normalized_matrix = normalize_gram_matrix(covariance_matrix)

    #     # Assert diagonals are 1
    #     assert_array_almost_equal(
    #         np.diagonal(normalized_matrix), np.array([1.0, 1.0, 1.0, 1.0])
    #     )

    #     # Test 1st row
    #     row = 0
    #     row_mul = row + 3
    #     multiplier = np.sqrt(
    #         np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
    #     )
    #     truth_array = covariance_matrix[row] / multiplier
    #     assert_array_almost_equal(normalized_matrix[row], truth_array)

    #     # Test 2nd row
    #     row = 1
    #     row_mul = row + 3
    #     multiplier = np.sqrt(
    #         np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
    #     )
    #     truth_array = covariance_matrix[row] / multiplier
    #     assert_array_almost_equal(normalized_matrix[row], truth_array)

    #     # Test 3rd row
    #     row = 2
    #     row_mul = row + 3
    #     multiplier = np.sqrt(
    #         np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
    #     )
    #     truth_array = covariance_matrix[row] / multiplier
    #     assert_array_almost_equal(normalized_matrix[row], truth_array)

    #     # Test 4th row
    #     row = 3
    #     row_mul = row + 3
    #     multiplier = np.sqrt(
    #         np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
    #     )
    #     truth_array = covariance_matrix[row] / multiplier
    #     assert_array_almost_equal(normalized_matrix[row], truth_array)

    # def test_compute_magnitude_density(self):
    #     """
    #     Test that the magnitude density is correctly computed.

    #         * Compute a gram matrix
    #         * Compute magnitude density
    #         * Compare to norm of vectors
    #     """
    #     # Create a random array
    #     array = onp.random.random((7, 10))
    #     # Compute a scalar product matrix of the array
    #     matrix = np.einsum("ij, kj -> ik", array, array)
    #     # Compute the density of array amplitudes
    #     array_norm = onp.linalg.norm(array, ord=2, axis=1)
    #     array_norm_density = array_norm / array_norm.sum()

    #     # Evaluate the magnitude density with the function that is to be tested
    #     mag_density = compute_magnitude_density(matrix)

    #     assert_array_almost_equal(array_norm_density, mag_density)

    # def test_flatten_rank_4_tensor(self):
    #     """
    #     Test the flattening of a rank 4 tensor.
    #     """
    #     # Check for assertion errors
    #     tensor = np.arange(24).reshape((2, 3, 2, 2))
    #     assert_raises(ValueError, flatten_rank_4_tensor, tensor)
    #     tensor = np.arange(24).reshape((2, 2, 3, 2))
    #     assert_raises(ValueError, flatten_rank_4_tensor, tensor)

    #     # Check the flattening
    #     tensor = np.arange(4 * 4).reshape(2, 2, 2, 2)
    #     assertion_matrix = np.array(
    #         [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    #     )
    #     assert_array_equal(flatten_rank_4_tensor(tensor), assertion_matrix)
