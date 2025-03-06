import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
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

class MLPRegression(BaseSolver):
    """
    A non-linear surrogate model that uses an MLP regressor to capture non-linear
    interactions in the ablation data. It computes local attributions by estimating
    the gradient of the model's prediction at the full-context baseline (a vector
    of ones) using finite differences.
    """
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (20, 10),
        activation: str = "tanh",
        solver: str = "lbfgs",
        alpha: float = 1e-4,
        max_iter: int = 2000,
        epsilon: float = 1e-5,
        random_state: int = 0,
    ) -> None:
        """
        Args:
            hidden_layer_sizes: Size of each hidden layer in the MLP.
            activation: Activation function ('tanh', 'relu', etc.).
            solver: Optimization solver for the MLP ('lbfgs', 'adam', etc.).
            alpha: L2 penalty (regularization term).
            max_iter: Maximum number of iterations.
            epsilon: Small finite difference used for gradient approximation.
            random_state: Random seed for reproducibility.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, float]:
        """
        Fits an MLP regressor to the ablation data and computes a local linear
        approximation (gradient) at a baseline (the full-context mask) via finite differences.

        Parameters:
            masks (NDArray): The binary ablation masks.
            outputs (NDArray): Aggregated logit-probabilities (one value per mask).
            num_output_tokens (int): Used to scale outputs back to original scale.

        Returns:
            Tuple[NDArray, float]: (grad, bias) where:
                - grad is an array of attribution scores (one per source).
                - bias is a scalar intercept term.
        """
        # Prepare data: convert masks to float and normalize outputs.
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens

        # Standardize features.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the MLP regressor with given hyperparameters.
        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        mlp.fit(X_scaled, Y)

        # Baseline: full-context mask (all ones).
        n_features = X.shape[1]
        x0 = np.ones((1, n_features), dtype=np.float32)
        x0_scaled = scaler.transform(x0)
        base_val = mlp.predict(x0_scaled)[0]

        # Compute gradient at the baseline via finite differences.
        grad = np.zeros(n_features, dtype=np.float32)
        for i in range(n_features):
            x_plus = x0.copy()
            x_plus[0, i] += self.epsilon
            x_plus_scaled = scaler.transform(x_plus)
            val_plus = mlp.predict(x_plus_scaled)[0]
            grad[i] = (val_plus - base_val) / self.epsilon

        # Compute bias such that f(x0) = grad · x0 + bias.
        bias = base_val - grad.dot(x0[0])

        # Scale attributions back to original output scale.
        return grad * num_output_tokens, bias * num_output_tokens