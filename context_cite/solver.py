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


# A helper wrapper to apply the polynomial transformation during prediction.
class PolyWeightWrapper:
    def __init__(self, weight: NDArray, poly: PolynomialFeatures):
        self.weight = weight
        self.poly = poly

    def __rmatmul__(self, other: NDArray) -> NDArray:
        """
        When the original masks (other) are multiplied on the left,
        first transform them into the polynomial feature space and then
        multiply by the weight vector.
        """
        X_poly = self.poly.transform(other.astype(np.float32))
        return X_poly @ self.weight


class PolynomialLassoRegression(BaseSolver):
    """
    A LASSO solver with polynomial feature expansion to capture non-linear interactions.
    
    Attributes:
        lasso_alpha (float): The alpha parameter for LASSO regression.
        degree (int): The degree of polynomial features to generate.
    """

    def __init__(self, lasso_alpha: float = 0.01, degree: int = 2) -> None:
        self.lasso_alpha = lasso_alpha
        self.degree = degree

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens
        # Expand features to include interaction terms
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        scaler = StandardScaler()
        lasso = Lasso(alpha=self.lasso_alpha, random_state=0, fit_intercept=True)
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X_poly, Y)
        
        # Compute coefficients on polynomial feature space.
        weight = lasso.coef_ / scaler.scale_
        bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T

        # Wrap the weight so that when used with the original masks,
        # the polynomial transformation is applied automatically.
        wrapped_weight = PolyWeightWrapper(weight * num_output_tokens, poly)
        return wrapped_weight, bias * num_output_tokens
