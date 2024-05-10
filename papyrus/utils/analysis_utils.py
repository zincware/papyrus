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

from papyrus.utils.matrix_utils import compute_hermitian_eigensystem


def compute_trace(matrix: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute the trace of a matrix, including optional normalization.

    Parameters
    ----------
    matrix : np.ndarray
            Matrix to calculate the trace of.
    normalize : bool (default=True)
            If true, the trace is normalized by the size of the matrix.

    Returns
    -------
    trace : np.ndarray
            Trace of the matrix.
    """
    normalization_factor = np.shape(matrix)[0] if normalize else 1
    return np.trace(matrix) / normalization_factor


def compute_shannon_entropy(dist: np.ndarray, normalize: bool = False) -> float:
    """
    Compute the Shannon entropy of a given probability distribution.

    The Shannon entropy of a given probability distribution is computed using a
    mask to neglect encountered zeros in the logarithm.

    Parameters
    ----------
    dist : np.ndarray
            Array to calculate the entropy of.
    normalize : bool (default = False)
            If true, the Shannon entropy is normalized by re-scaling to the maximum
            entropy. The method will return a value between 0 and 1.

    Returns
    -------
    Entropy of the distribution
    """
    mask = np.nonzero(dist)
    scaled_values = -1 * dist[mask] * np.log(dist[mask])
    entropy = scaled_values.sum()

    if normalize:
        scale_factor = np.log(len(dist))
        entropy /= scale_factor

    return entropy


def compute_von_neumann_entropy(
    matrix: np.ndarray, effective: bool = True, normalize_eig: bool = True
) -> float:
    """
    Compute the von-Neumann entropy of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
            Matrix for which the entropy should be computed.
    effective : bool (default=True)
            If true, the entropy is divided by the theoretical maximum entropy of
            the system thereby returning the effective entropy.
    normalize_eig : bool (default = True)
            If true, the eigenvalues are scaled to look like probabilities.

    Returns
    -------
    entropy : float
            Von-Neumann entropy of the matrix.
    """
    eigvals, _ = compute_hermitian_eigensystem(matrix, normalize=normalize_eig)

    entropy = compute_shannon_entropy(eigvals)

    if effective:
        maximum_entropy = np.log(len(eigvals))
        entropy /= maximum_entropy

    return entropy
