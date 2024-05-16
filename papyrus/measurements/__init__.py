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

from papyrus.measurements.base_measurement import BaseMeasurement
from papyrus.measurements.measurements import (
    NTK,
    Accuracy,
    Loss,
    NTKEigenvalues,
    NTKEntropy,
    NTKMagnitudeDistribution,
    NTKSelfEntropy,
    NTKTrace,
)

__all__ = [
    BaseMeasurement.__name__,
    NTKTrace.__name__,
    NTKEntropy.__name__,
    NTKSelfEntropy.__name__,
    NTKEigenvalues.__name__,
    NTKMagnitudeDistribution.__name__,
    Loss.__name__,
    Accuracy.__name__,
    NTK.__name__,
]
