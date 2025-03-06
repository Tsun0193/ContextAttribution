import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline


class BaseSolver(ABC):
    """
    A base solver class.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    @abstractmethod
    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        pass


class LassoRegression(BaseSolver):
    """
    A LASSO solver using the scikit-learn library.
    """

    def __init__(self, lasso_alpha: float = 0.01) -> None:
        self.lasso_alpha = lasso_alpha

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens
        scaler = StandardScaler()
        lasso = Lasso(alpha=self.lasso_alpha, random_state=0, fit_intercept=True)
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X, Y)
        # Rescale back to original scale
        weight = lasso.coef_ / scaler.scale_
        bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
        return weight * num_output_tokens, bias * num_output_tokens

class PolynomialLassoRegression(BaseSolver):
    """
    A non-linear surrogate model that uses a polynomial feature expansion
    followed by LASSO regression. It computes attributions by locally
    approximating the model with a linear function via finite differences.
    """
    def __init__(self, lasso_alpha: float = 0.01, degree: int = 2, epsilon: float = 1e-5) -> None:
        self.lasso_alpha = lasso_alpha
        self.degree = degree
        self.epsilon = epsilon

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        """
        Fits a polynomial LASSO regression model to the ablation data and computes
        a local linear approximation (gradient) at the full-context baseline.

        Parameters:
            masks (NDArray): Ablation masks (binary indicators for each source).
            outputs (NDArray): Aggregated logit-probabilities.
            num_output_tokens (int): Used for scaling the outputs.

        Returns:
            Tuple[NDArray, NDArray]: A tuple (weight, bias) where weight is an array
            containing the attribution score for each source and bias is a scalar.
        """
        # Prepare data: scale the outputs similar to the linear model.
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens

        # Create a pipeline: standardize, expand features polynomially, then apply LASSO.
        scaler = StandardScaler()
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        lasso = Lasso(alpha=self.lasso_alpha, random_state=0, fit_intercept=True)
        pipeline = make_pipeline(scaler, poly, lasso)
        pipeline.fit(X, Y)

        # Define the baseline: the full-context mask (all ones).
        n_features = X.shape[1]
        x0 = np.ones(n_features, dtype=np.float32)

        # Compute the prediction at the baseline.
        base_val = pipeline.predict(x0.reshape(1, -1))[0]

        # Compute the gradient (finite differences) to obtain local attributions.
        grad = np.zeros_like(x0, dtype=np.float32)
        for i in range(n_features):
            x_plus = x0.copy()
            x_plus[i] += self.epsilon
            val_plus = pipeline.predict(x_plus.reshape(1, -1))[0]
            grad[i] = (val_plus - base_val) / self.epsilon

        # Compute a local linear approximation intercept.
        bias = base_val - grad.dot(x0)

        # Scale the attributions back to the original output scale.
        return grad * num_output_tokens, bias * num_output_tokens
