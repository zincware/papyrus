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
        apply_fn : Optional[Callable] (default=None)
                The loss function to be used to compute the loss of the neural network.
                If the loss function is not provided, the apply method will assume that
                the loss is used as the input.
                If the loss function is provided, the apply method will assume that the
                neural network outputs and the target values are used as inputs.
        """
        super().__init__(name, rank)
        self.apply_fn = apply_fn

    def apply(
        self,
        loss: Optional[float] = None,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ) -> float:
        """
        Method to record the loss of a neural network.

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        loss : Optional[float] (default=None)
                The loss of the neural network.
        predictions : Optional[np.ndarray] (default=None)
                The predictions of the neural network.
        targets : Optional[np.ndarray] (default=None)
                The target values of the neural network.

        Returns
        -------
        loss : float
            The loss of the neural network.
        """
        # Check if any of the inputs are None
        if loss is None and (predictions is None or targets is None):
            raise ValueError(
                "Either the loss or the predictions and targets must be provided."
            )
        # Check if a loss value and the predictions and targets are provided
        if loss is not None and (predictions is not None or targets is not None):
            raise ValueError(
                "Either the loss or the predictions and targets must be provided."
            )
        # If the loss is provided, return the loss
        if loss is not None:
            return loss
        # If the loss is not provided, compute the loss using the loss function
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
        apply_fn : Optional[Callable] (default=None)
                The accuracy function to be used to compute the accuracy of the neural
                network.
                # If the accuracy function is not provided, the apply method will assume
                that the accuracy is used as the input.
                If the accuracy function is provided, the apply method will assume that
                the neural network outputs and the target values are used as inputs.
        """
        super().__init__(name, rank)
        self.apply_fn = apply_fn

    def apply(
        self,
        accuracy: Optional[float] = None,
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
        # Check if any of the inputs are None
        if accuracy is None and (predictions is None or targets is None):
            raise ValueError(
                "Either the accuracy or the predictions and targets must be provided."
            )
        # Check if a loss value and the predictions and targets are provided
        if accuracy is not None and (predictions is not None or targets is not None):
            raise ValueError(
                "Either the accuracy or the predictions and targets must be provided."
            )
        # If the accuracy is provided, return the accuracy
        if accuracy is not None:
            return accuracy
        # If the accuracy is not provided, compute the accuracy using the accuracy
        # function
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
        super().__init__(name, rank)
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
        normalize_eigenvalues : bool (default=True)
                If true, the eigenvalues are scaled to look like probabilities.
        effective : bool (default=False)
                If true, the entropy is divided by the theoretical maximum entropy of
                the system thereby returning the effective entropy / entropy density.

        """
        super().__init__(name=name, rank=rank)
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
        effective : bool (default=False)
                Boolean flag to indicate whether the self-entropy of the NTK will be
                normalized by the theoretical maximum entropy of the system.
        """
        super().__init__(name, rank)
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
        super().__init__(name, rank)

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
        normalize : bool (default=True)
                Boolean flag to indicate whether the eigenvalues of the NTK will be
                normalized by the size of the NTK matrix.
        """
        super().__init__(name, rank)
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
        """
        super().__init__(name, rank)

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


class LossDerivative(BaseMeasurement):
    """
    Measurement class to record the derivative of the loss with respect to the neural
    network outputs.

    Neural State Keys
    -----------------
    loss_derivative : np.ndarray
            The derivative of the loss with respect to the weights.
    """

    def __init__(
        self,
        apply_fn: Callable,
        name: str = "loss_derivative",
        rank: int = 1,
    ):
        """
        Constructor method of the LossDerivative class.

        Parameters
        ----------
        apply_fn : Callable
                The function to compute the derivative of the loss with respect to the
                neural network outputs.
        name : str (default="loss_derivative")
                The name of the measurement, defining how the instance in the database
                will be identified.
        rank : int (default=1)
                The rank of the measurement, defining the tensor order of the
                measurement.
        """
        super().__init__(name, rank)
        self.apply_fn = apply_fn

    def apply(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Method to record the derivative of the loss with respect to the neural network
        outputs.

        Parameters need to be provided as keyword arguments.

        Parameters
        ----------
        predictions : np.ndarray
                The predictions of the neural network.
        targets : np.ndarray
                The target values of the neural network.

        Returns
        -------
        np.ndarray
            The derivative of the loss with respect to the neural network outputs.
        """
        return self.apply_fn(predictions, targets)
