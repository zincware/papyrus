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

from papyrus.utils.analysis_utils import (
    compute_shannon_entropy,
    compute_trace,
    compute_von_neumann_entropy,
)
from papyrus.utils.matrix_utils import (
    compute_gramian_diagonal_distribution,
    compute_hermitian_eigensystem,
    compute_l_pq_norm,
    compute_matrix_alignment,
    compute_vector_outer_product,
    flatten_rank_4_tensor,
    normalize_gram_matrix,
    unflatten_rank_4_tensor,
)

__all__ = [
    compute_hermitian_eigensystem.__name__,
    normalize_gram_matrix.__name__,
    compute_gramian_diagonal_distribution.__name__,
    compute_l_pq_norm.__name__,
    flatten_rank_4_tensor.__name__,
    unflatten_rank_4_tensor.__name__,
    compute_trace.__name__,
    compute_shannon_entropy.__name__,
    compute_von_neumann_entropy.__name__,
    compute_vector_outer_product.__name__,
    compute_matrix_alignment.__name__,
]
