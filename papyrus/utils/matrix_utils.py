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
Matrix utils for papyrus.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_hermitian_eigensystem(
    matrix: np.ndarray, normalize: bool = False, clip: bool = True
):
    """
    Compute the eigenspace of a hermitian matrix.

    Parameters
    ----------
    matrix : np.ndarray
            Matrix for which the space should be computed. Must be hermitian.
    normalize : bool (default=False)
            If true, the eigenvalues are divided by the sum of the eigenvalues of the
            given matrix. This is equivalent to dividing by the size of the dataset.
    clip : bool (default=True)
            Clip the eigenvalues to a very small number to avoid negatives. This should
            only be used if you are sure that negative numbers only arise due to some
            numeric reason and that they should not exist.

    Returns
    -------
    eigenvalues : np.ndarray
            Eigenvalues of the matrix. The eigenvalues are sorted in descending order.
    eigenvectors : np.ndarray
            Eigenvectors of the matrix.
            The column `eigenvectors[:, i]` is the normalized eigenvector corresponding
            to the eigenvalue `eigenvalues[i]`.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    if clip:
        # logger.warning("Eigenvalues are being clipped to avoid negative values.")
        eigenvalues = np.clip(eigenvalues, 1e-14, None)

    if normalize:
        eigenvalues /= eigenvalues.sum()

    return eigenvalues[::-1], eigenvectors[:, ::-1]


def normalize_gram_matrix(gram_matrix: np.ndarray):
    """
    Method for normalizing a gram matrix.

    The normalization is done by dividing each element of the gram matrix by the
    square root of the product of the corresponding diagonal elements. This is
    equivalent to normalizing the inputs of the gram matrix.

    Parameters
    ----------
    gram_matrix : np.ndarray
            Gram matrix to normalize.

    Returns
    -------
    normalized_gram_matrix : np.ndarray
            A normalized gram matrix, i.e, the matrix given if all of its inputs
            had been normalized.
    """
    order = np.shape(gram_matrix)[0]

    diagonals = np.diagonal(gram_matrix)

    repeated_diagonals = np.repeat(diagonals[None, :], order, axis=0)

    normalizing_matrix = np.sqrt(repeated_diagonals * repeated_diagonals.T)

    # Check for zeros in the normalizing matrix
    normalizing_matrix = np.where(
        normalizing_matrix == 0, 0, 1 / normalizing_matrix
    )  # Avoid division by zero

    return gram_matrix * normalizing_matrix


def compute_gramian_diagonal_distribution(gram_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the normalized distribution of the diagonals of a gram matrix.

    The distribution is computed by taking the square root of the diagonal elements
    of the gram matrix and normalizing them by the sum.
    This method is equivalent to the distribution of the magnitudes of the vectors
    that were used to compute the gram matrix.

    Parameters
    ----------
    gram_matrix : np.ndarray
            Gram matrix to compute the diagonal distribution of.

    Returns
    -------
    magnitude_density: np.ndarray
            Magnitude density of the individual entries.
    """
    magnitudes = np.sqrt(np.diagonal(gram_matrix))
    distribution = magnitudes / magnitudes.sum()
    return distribution


def compute_l_pq_norm(matrix: np.ndarray, p: int = 2, q: int = 2):
    """
    Compute the L_pq norm of a matrix.

    The norm calculates (sum_j (sum_i abs(a_ij)^p)^(p/q) )^(1/q)
    For the defaults (p = 2, q = 2) the function calculates the Frobenius
    (or Hilbert-Schmidt) norm.

    Parameters
    ----------
    matrix: np.ndarray
            Matrix to calculate the L_pq norm of
    p: int (default=2)
            Inner power of the norm.
    q: int (default=2)
            Outer power of the norm.

    Returns
    -------
    calculate_l_pq_norm: np.ndarray
            L_qp norm of the matrix.
    """
    inner_sum = np.sum(np.power(np.abs(matrix), p), axis=-1)
    outer_sum = np.sum(np.power(inner_sum, q / p), axis=-1)
    return np.power(outer_sum, 1 / q)


def compute_vector_outer_product(vector: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of a vector.

    The outer product is computed as the product of the vector with its transpose.
    Note: Providing a vector of rank 2 will result in an outer product of rank 4.
    Example:
    vector = np.array([ [0, 1], [1, 0] ])
    outer_product = np.array([[
        [[0, 0],
         [0, 1]],

        [[0, 0],
         [1, 0]]],

       [[[0, 1],
         [0, 0]],

        [[1, 0],
         [0, 0]]
    ]])

    Parameters
    ----------
    vector : np.ndarray
            Vector to compute the outer product of. Must be of rank 1 or 2.

    Returns
    -------
    outer_product : np.ndarray
            Outer product of the vector.
    """
    shape = vector.shape
    product = np.outer(vector, vector)
    if len(shape) == 2:
        return unflatten_rank_4_tensor(
            product, (shape[0], shape[0], shape[1], shape[1])
        )
    elif len(shape) == 1:
        return product
    else:
        raise ValueError(
            "The input to compute the outer product is not of rank 1 or 2. "
            f"Expected rank 1 or 2 but got {len(shape)}."
        )


def compute_matrix_alignment(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    """
    Compute the alignment between two matrices.

    The alignment is computed as the cosine similarity between the flattened
    matrices. This is equivalent to the cosine similarity between the vectors
    obtained by flattening the matrices.

    Parameters
    ----------
    matrix_1 : np.ndarray
            First matrix to compute the alignment of.
    matrix_2 : np.ndarray
            Second matrix to compute the alignment of.

    Returns
    -------
    alignment : float
            Alignment between the two matrices.
    """
    dot_product = np.sum(matrix_1 * matrix_2)
    norm_1 = np.linalg.norm(matrix_1)
    norm_2 = np.linalg.norm(matrix_2)
    alignment = dot_product / (norm_1 * norm_2)
    return alignment


def flatten_rank_4_tensor(tensor: np.ndarray) -> tuple:
    """
    Flatten a rank 4 tensor to a rank 2 tensor using a specific reshaping.

    With this function a tensor of shape (k, l, m, n) is reshaped to a tensor of shape
    (k * m, l * n). The reshaping is done by concatenating axes 1 and 2, and
    then 3 and 4.

    Parameters
    ----------
    tensor : np.ndarray (shape=(k, l, m, n))
            Tensor to flatten.

    Returns
    -------
    A 2-tuple of the following form:

    flattened_tensor : np.ndarray (shape=(k * m, l * n))
            Flattened tensor.
    shape : tuple
            Shape of the original tensor.
    """
    # Check if the tensor is of rank 4
    if len(tensor.shape) != 4:
        raise ValueError(
            "The tensor is not of rank 4. "
            f"Expected rank 4 but got {len(tensor.shape)}."
        )
    shape = tensor.shape
    _tensor = np.moveaxis(tensor, [1, 2], [2, 1])
    flattened_tensor = _tensor.reshape(shape[0] * shape[2], shape[1] * shape[3])
    return flattened_tensor, shape


def unflatten_rank_4_tensor(tensor: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Unflatten a rank 2 tensor to a rank 4 tensor using a specific reshaping.

    This is the inverse operation of the flatten_rank_4_tensor function.
    The tensor is assumed to be of shape (k * m, l * n) and is reshaped to a tensor
    of shape (k, l, m, n). The reshaping is done by splitting the first axis into
    two axes, and then the second axis into two axes.

    Parameters
    ----------
    tensor : np.ndarray (shape=(k * m, l * n))
            Tensor to unflatten.
    shape : tuple
            Shape of the original tensor. Must be of rank 4.

    Returns
    -------
    unflattened_tensor : np.ndarray (shape=(k, l, m, n))
            Unflattened tensor.
    """
    # Check if the tensor is of rank 2
    if len(tensor.shape) != 2:
        raise ValueError(
            "The tensor is not of rank 2. "
            f"Expected rank 2 but got {len(tensor.shape)}."
        )
    # Check if the shape is of rank 4
    if len(new_shape) != 4:
        raise ValueError(
            "The shape is not of rank 4. " f"Expected rank 4 but got {len(new_shape)}."
        )
    # Check if the shapes match
    if not new_shape[0] * new_shape[2] == tensor.shape[0]:
        raise ValueError(
            "The shape of the tensor does not match the given dimensions. "
            f"Expected {new_shape[0] * new_shape[2]} but got {tensor.shape[0]}."
        )
    if not new_shape[1] * new_shape[3] == tensor.shape[1]:
        raise ValueError(
            "The shape of the tensor does not match the given dimensions. "
            f"Expected {new_shape[1] * new_shape[3]} but got {tensor.shape[1]}."
        )
    _tensor = tensor.reshape(new_shape[0], new_shape[2], new_shape[1], new_shape[3])
    return np.moveaxis(_tensor, [2, 1], [1, 2])
