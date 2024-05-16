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
Module containing default measurements for recording neural learning.
"""

from typing import Callable, Optional

import numpy as np

from papyrus.measurements.base_measurement import BaseMeasurement
from papyrus.utils.analysis_utils import (
    compute_shannon_entropy,
    compute_trace,
    compute_von_neumann_entropy,
)
from papyrus.utils.matrix_utils import (
    compute_grammian_diagonal_distribution,
    compute_hermitian_eigensystem,
)


class Loss(BaseMeasurement):
    """
    Measurement class to record the loss of a neural network.

    Neural State Keys
    -----------------
    predictions : np.ndarray
            The predictions of the neural network. Required if the loss function is
            provided. Needs to be combined with the targets key.
    targets : np.ndarray
            The target values of the neural network. Required if the loss function is
            provided. Needs to be combined with the predictions key.
    loss : float
            The loss of the neural network. Required if the loss function is not
            provided. Allows to measure precomputed loss values.
    """

    def __init__(
        self,
        name: str = "loss",
        rank: int = 0,
        public: bool = False,
        apply_fn: Optional[Callable] = None,
    ):
        """
        Constructor method of the Loss class.

        Parameters
        ----------
        name : str (default="loss")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=0)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        apply_fn : Optional[Callable] (default=None)
                The loss function to be used to compute the loss of the neural network.
                If the loss function is not provided, the apply method will assume that
                the loss is used as the input.
                If the loss function is provided, the apply method will assume that the
                neural network outputs and the target values are used as inputs.
        """
        super().__init__(name, rank, public)

        self.apply_fn = apply_fn

        # Based on the provided loss function, set the apply method
        if self.apply_fn is None:
            self.apply = self._apply_no_computation
        else:
            self.apply = self._apply_computation

        self.neural_state_keys = self._get_apply_signature()

    def _apply_no_computation(
        self,
        loss: Optional[float] = None,
    ) -> float:
        """
        Method to record the loss of a neural network.

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        loss : Optional[float] (default=None)
                The loss of the neural network.

        Returns
        -------
        loss : float
            The loss of the neural network.
        """
        return loss

    def _apply_computation(
        self,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ) -> float:
        """
        Method to record the loss of a neural network.

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        predictions : Optional[np.ndarray] (default=None)
                The predictions of the neural network.
        targets : Optional[np.ndarray] (default=None)
                The target values of the neural network.

        Returns
        -------
        loss : float
            The loss of the neural network.
        """
        return self.apply_fn(predictions, targets)


class Accuracy(BaseMeasurement):
    """
    Measurement class to record the accuracy of a neural network.

    Neural State Keys
    -----------------
    predictions : np.ndarray
            The predictions of the neural network. Required if the loss function is
            provided. Needs to be combined with the targets key.
    targets : np.ndarray
            The target values of the neural network. Required if the loss function is
            provided. Needs to be combined with the predictions key.
    accuracy : float
            The accuracy of the neural network. Required if the accuracy function is not
            provided. Allows to measure precomputed loss values.
    """

    def __init__(
        self,
        name: str = "accuracy",
        rank: int = 0,
        public: bool = False,
        apply_fn: Optional[Callable] = None,
    ):
        """
        Constructor method of the Accuracy class.

        Parameters
        ----------
        name : str (default="accuracy")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=0)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        apply_fn : Optional[Callable] (default=None)
                The accuracy function to be used to compute the accuracy of the neural
                network.
                # If the accuracy function is not provided, the apply method will assume
                that the accuracy is used as the input.
                If the accuracy function is provided, the apply method will assume that
                the neural network outputs and the target values are used as inputs.
        """
        super().__init__(name, rank, public)

        self.apply_fn = apply_fn

        # Based on the provided accuracy function, set the apply method
        if self.apply_fn is None:
            self.apply = self._apply_no_computation
        else:
            self.apply = self._apply_computation

        self.neural_state_keys = self._get_apply_signature()

    def _apply_no_computation(self, accuracy: Optional[float] = None) -> float:
        """
        Method to record the accuracy of a neural network.

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        accuracy : Optional[float] (default=None)
                The accuracy of the neural network.

        Returns
        -------
        accuracy : float
            The accuracy of the neural network.
        """
        return accuracy

    def _apply_computation(
        self,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ) -> float:
        """
        Method to record the accuracy of a neural network.

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        accuracy : Optional[float] (default=None)
                The accuracy of the neural network.
        predictions : Optional[np.ndarray] (default=None)
                The predictions of the neural network.
        targets : Optional[np.ndarray] (default=None)
                The target values of the neural network.

        Returns
        -------
        accuracy : float
            The accuracy of the neural network.
        """
        return self.apply_fn(predictions, targets)


class NTKTrace(BaseMeasurement):
    """
    Measurement class to record the trace of the NTK.

    Neural State Keys
    -----------------
    ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
    """

    def __init__(
        self,
        name: str = "ntk_trace",
        rank: int = 1,
        public: bool = False,
        normalize: bool = True,
    ):
        """
        Constructor method of the NTKTrace class.

        Parameters
        ----------
        name : str (default="ntk_trace")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=1)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        normalize : bool (default=True)
                Boolean flag to indicate whether the trace of the NTK will be normalized
                by the size of the NTK matrix.
        """
        super().__init__(name, rank, public)
        self.normalise = normalize

    def apply(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to compute the trace of the NTK.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.

        Returns
        -------
        np.ndarray
            The trace of the NTK
        """
        if ntk.shape[0] != ntk.shape[1]:
            raise ValueError(
                "To compute the trace of the NTK, the NTK matrix must"
                f" be a square matrix, but got a matrix of shape {ntk.shape}."
            )
        if len(ntk.shape) != 2:
            raise ValueError(
                "To compute the trace of the NTK, the NTK matrix must"
                f" be a tensor of rank 2, but got a tensor of rank {len(ntk.shape)}."
            )
        return compute_trace(ntk, normalize=self.normalise)


class NTKEntropy(BaseMeasurement):
    """
    Measurement class to record the entropy of the NTK.

    Neural State Keys
    -----------------
    ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
    """

    def __init__(
        self,
        name: str = "ntk_cross_entropy",
        rank: int = 1,
        public: bool = False,
        normalize_eigenvalues: bool = True,
        effective: bool = False,
    ):
        """
        Constructor method of the NTKCrossEntropy class.

        Parameters
        ----------
        name : str (default="ntk_trace")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=1)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        normalize_eigenvalues : bool (default=True)
                If true, the eigenvalues are scaled to look like probabilities.
        effective : bool (default=False)
                If true, the entropy is divided by the theoretical maximum entropy of
                the system thereby returning the effective entropy / entropy density.

        """
        super().__init__(name=name, rank=rank, public=public)
        self.normalize_eigenvalues = normalize_eigenvalues
        self.effective = effective

    def apply(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to compute the entropy of the NTK.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
            Note that the NTK matrix needs to be a 2D square matrix. In case of a
            4D NTK tensor, please flatten the tensor to a 2D matrix before passing
            it to this method. A default implementation of a flattenig method is
            provided in the papyrus.utils.matrix_utils module as
            flatten_rank_4_tensor.

        Returns
        -------
        np.ndarray
            The entropy of the NTK.
        """
        # Assert that the NTK is a square matrix
        if ntk.shape[0] != ntk.shape[1]:
            raise ValueError(
                "To compute the entropy of the NTK, the NTK matrix must"
                f" be a square matrix, but got a matrix of shape {ntk.shape}."
            )
        if len(ntk.shape) != 2:
            raise ValueError(
                "To compute the entropy of the NTK, the NTK matrix must"
                f" be a tensor of rank 2, but got a tensor of rank {len(ntk.shape)}."
            )
        # Compute the von Neumann entropy of the NTK
        return compute_von_neumann_entropy(
            ntk,
            effective=self.effective,
            normalize_eig=self.normalize_eigenvalues,
        )


class NTKSelfEntropy(BaseMeasurement):
    """
    Measurement class to record the entropy of the diagonal of the NTK.

    This measurement can be interpreted as the entropy of the self-correlation of data
    in a neural network.

    Neural State Keys
    -----------------
    ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
    """

    def __init__(
        self,
        name: str = "ntk_self_entropy",
        rank: int = 0,
        public: bool = False,
        effective: bool = False,
    ):
        """
        Constructor method of the NTKSelfEntropy class.

        Parameters
        ----------
        name : str (default="ntk_magnitude_distribution")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=1)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        effective : bool (default=False)
                Boolean flag to indicate whether the self-entropy of the NTK will be
                normalized by the theoretical maximum entropy of the system.
        """
        super().__init__(name, rank, public)
        self.effective = effective

    def apply(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to compute the self-entropy of the NTK.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.

        Returns
        -------
        np.ndarray
            Self-entropy of the NTK.
        """
        if ntk.shape[0] != ntk.shape[1]:
            raise ValueError(
                "To compute the self-entropy of the NTK, the NTK matrix must"
                f" be a square matrix, but got a matrix of shape {ntk.shape}."
            )
        if len(ntk.shape) != 2:
            raise ValueError(
                "To compute the self-entropy of the NTK, the NTK matrix must"
                f" be a tensor of rank 2, but got a tensor of rank {len(ntk.shape)}."
            )
        distribution = compute_grammian_diagonal_distribution(gram_matrix=ntk)
        return compute_shannon_entropy(distribution, effective=self.effective)


class NTKMagnitudeDistribution(BaseMeasurement):
    """
    Measurement class to record the magnitude distribution of the NTK.

    Note
    ----
    This measurement is not applicable to varying number of inputs as its output size
    depends on the number of inputs it is applied to.

    Measurements that return arrays with sizes that depend on the number of inputs
    **cannot** be applied on varying number of inputs. This is because the number of
    dimensions of the input need to be same for all subsequent calls, otherwise an error
    will be raised when storing the results in the database.


    Neural State Keys
    -----------------
    ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
    """

    def __init__(
        self,
        name: str = "ntk_magnitude_distribution",
        rank: int = 0,
        public: bool = False,
    ):
        """
        Constructor method of the NTKMagnitudeDistribution class.

        Parameters
        ----------
        name : str (default="ntk_magnitude_distribution")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=1)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        """
        super().__init__(name, rank, public)

    def apply(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to compute the magnitude distribution of the NTK.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.

        Returns
        -------
        np.ndarray
            The magnitude distribution of the NTK
        """
        if ntk.shape[0] != ntk.shape[1]:
            raise ValueError(
                "To compute the magnitude distribution of the NTK, the NTK matrix must"
                f" be a square matrix, but got a matrix of shape {ntk.shape}."
            )
        if len(ntk.shape) != 2:
            raise ValueError(
                "To compute the magnitude distribution of the NTK, the NTK matrix must"
                f" be a tensor of rank 2, but got a tensor of rank {len(ntk.shape)}."
            )
        return compute_grammian_diagonal_distribution(gram_matrix=ntk)


class NTKEigenvalues(BaseMeasurement):
    """
    Measurement class to record the eigenvalues of the NTK.

    Note
    ----
    This measurement is not applicable to varying number of inputs as its output size
    depends on the number of inputs it is applied to.

    Measurements that return arrays with sizes that depend on the number of inputs
    **cannot** be applied on varying number of inputs. This is because the number of
    dimensions of the input need to be same for all subsequent calls, otherwise an error
    will be raised when storing the results in the database.

    Neural State Keys
    -----------------
    ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
    """

    def __init__(
        self,
        name: str = "ntk_eigenvalues",
        rank: int = 1,
        public: bool = False,
        normalize: bool = True,
    ):
        """
        Constructor method of the NTKEigenvalues class.

        Parameters
        ----------
        name : str (default="ntk_eigenvalues")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=1)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        normalize : bool (default=True)
                Boolean flag to indicate whether the eigenvalues of the NTK will be
                normalized by the size of the NTK matrix.
        """
        super().__init__(name, rank, public)
        self.normalize = normalize

    def apply(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to compute the eigenvalues of the NTK.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.

        Returns
        -------
        np.ndarray
            The eigenvalues of the NTK
        """
        if ntk.shape[0] != ntk.shape[1]:
            raise ValueError(
                "To compute the eigenvalues of the NTK, the NTK matrix must"
                f" be a square matrix, but got a matrix of shape {ntk.shape}."
            )
        if len(ntk.shape) != 2:
            raise ValueError(
                "To compute the eigenvalues of the NTK, the NTK matrix must"
                f" be a tensor of rank 2, but got a tensor of rank {len(ntk.shape)}."
            )
        return compute_hermitian_eigensystem(ntk, normalize=self.normalize)[0]


class NTK(BaseMeasurement):
    """
    Measurement class to record the Neural Tangent Kernel (NTK).

    Note
    ----
    This measurement is not applicable to varying number of inputs as its output size
    depends on the number of inputs it is applied to.

    Measurements that return arrays with sizes that depend on the number of inputs
    **cannot** be applied on varying number of inputs. This is because the number of
    dimensions of the input need to be same for all subsequent calls, otherwise an error
    will be raised when storing the results in the database.

    Neural State Keys
    -----------------
    ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
    """

    def __init__(
        self,
        name: str = "ntk",
        rank: int = 2,
        public: bool = False,
    ):
        """
        Constructor method of the NTK class.

        Parameters
        ----------
        name : str (default="ntk")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=2)
                The rank of the measurement, defining the tensor order of the
                measurement.
        public : bool (default=False)
                Boolean flag to indicate whether the measurement resutls will be
                accessible via a public attribute of the recorder.
        """
        super().__init__(name, rank, public)

    def apply(self, ntk: np.ndarray) -> np.ndarray:
        """
        Method to record the Neural Tangent Kernel (NTK).

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        ntk : np.ndarray
            The Neural Tangent Kernel (NTK) matrix.

        Returns
        -------
        np.ndarray
            The Neural Tangent Kernel (NTK) matrix.
        """
        return ntk
