"""Implementation of the Bayesian Safety Validation method."""
import math
from itertools import combinations
from typing import Dict, Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from .acq_max import acq_max
from .sample_space import SampleSpace
from .utils import ensure_rng, logistic, logit, phi


class BayesianSafetyValidation:
    """Implementation of Bayesian safety validation.

    Args:
        param_space (Dict): The parameter space to search in.
        The keys are the parameter names and the values are tuples with the lower and upper bounds.

        n_init (int, optional): The number of initial points to sample. Defaults to 5.

        random_state ([type], optional): The random state to use. Defaults to None.

    Usage:
    .. code-block:: python

        # Define a black function
        def black_function(x):
            return x**2

        # Create a BayesianSafetyValidation object
        from adt_sim.safty_validation import BayesianSafetyValidation
        param_space = {'x': (0, 10), 'y': (0, 10)}
        bsv = BayesianSafetyValidation(param_space)

        # Initial suggestions
        suggestions = bsv.suggest()

        # Iterate 10 times
        for _ in range(10):
            evaluations = [black_function(suggestion) for suggestion in suggestions]
            bsv.rdefit(suggestions, evaluations)
            suggestions = bsv.suggest()

        # Show result
        bsv.falsification()

    .. code-block:: python

        # There is a API under discussion
        bsv = BayesianSafetyValidation(
            black_box_func=black_box_func,
            param_space={"x1": (-10, 10), "x2": (-10, 10)},
            n_iter=10,
            n_init=5,
        )

        bsv.run()

    """

    def __init__(
        self,
        param_space: Dict,
        n_init=5,
        random_state=None,
        lamb=0.1,
    ) -> None:
        self._random_state = ensure_rng(random_state)
        self._n_init = n_init
        self._space = SampleSpace(
            param_space, random_state=random_state, allow_duplicate_points=True
        )
        self._surrogate_model = None
        self._lamda = lamb

    def suggest(self) -> Sequence[Dict]:
        """Return suggestions which are the next points to evaluate.

        Returns:
            dict: A sequnce of dict stored parameter to evaluate. Like:
                [
                    {"param1": value1, "param2": value2},
                    {"param1": value1, "param2": value2},
                    ...
                ]
        """

        if self._space.empty:
            randoms = self._space.random_sample(self._n_init)
            return [self._space.array_to_params(random) for random in randoms]

        x1 = self._uncertianity_exploration()
        x2 = self._boundary_refinement()
        # x3 = self._failure_region_sampling()

        return [x1, x2]

    def refit(
        self,
        additional_suggestions: Sequence[Dict],
        additional_evaluations: Sequence[float],
    ) -> None:
        """Refit the surrogate model with new suggestions and evaluations.

        Args:
            suggestions (Sequence[Dict]): _description_
            reward (Sequence[float]): _description_
        """
        if self._surrogate_model is None:
            self._surrogate_model = GaussianProcessRegressor(
                kernel=kernels.Matern(length_scale=math.exp(-0.1), nu=0.5)
            )

        assert len(additional_suggestions) == len(additional_evaluations)
        for i in range(len(additional_suggestions)):
            self._space.register(
                additional_suggestions[i], 0.1 * logit(phi(additional_evaluations[i]))
            )
        params = self._space.params
        target = self._space.target

        print(f"params: {params}")
        print(f"target: {target}")

        self._surrogate_model.fit(params, target)

    def falsification(self, param_space: Dict = None) -> NDArray:
        """_summary_

        Args:
            param_space (Dict, optional): A sub param space to evaluate.
            It shoule be a sub space of the original param space.
            Defaults to None indicating the original param space.

        Returns:
            NDArray:
        """
        mean_prediction = self._surrogate_model.predict(self._space.params)

        space = self._space if param_space is None else param_space
        dim = len(space.keys)

        fig, axes = plt.subplots(dim, dim, sharex="row", sharey="col")

        for i, j in combinations(range(dim), 2):
            axes[i, j].scatter(
                self._space.params[:, i],
                self._space.params[:, j],
                c=mean_prediction,
                cmap="viridis",
            )

        for row in range(dim):
            for col in range(dim):
                if row >= col:
                    axes[row, col].set_visible(False)

        plt.show()

    def most_likely_failure_point(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        pass

    def failure_probability(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        pass

    # def _variance_func(self, x: NDArray) -> float:
    #     assert self._surrogate_model is not None
    #     assert self._surrogate_model.kernel_ is not None
    #     assert self._surrogate_model.X_train_ is not None

    #     K = self._surrogate_model.kernel_
    #     X_train = self._surrogate_model.X_train_
    #     return K(x) - K(x, X_train) @ np.linalg.inv(K(X_train)) @ K(X_train, x)

    # TODO: cache
    def _k_start_func(self, x: NDArray) -> NDArray:
        K = self._surrogate_model.kernel_
        X_train = self._surrogate_model.X_train_

        return K(X_train, x.reshape(-1, self._space.dim))

    def _mean_func(self, x: NDArray) -> NDArray:
        alpha = self._surrogate_model.alpha_
        mean = self._k_start_func(x).T @ alpha
        return mean.reshape(x.shape[0], -1)

    def _variance_func(self, x: NDArray) -> NDArray:
        K = self._surrogate_model.kernel_
        L = self._surrogate_model.L_

        k_start = self._k_start_func(x)
        v = np.linalg.solve(L, k_start)

        var = K(x.reshape(-1, self._space.dim)) - v.T @ v

        assert var.shape[0] == var.shape[1], f"Invalid shape: {var.shape}"
        return var.diagonal().reshape(x.shape[0], -1)

    def _std_deviation(self, x: NDArray) -> NDArray:
        return np.sqrt(self._variance_func(x))

    def _uncertianity_exploration(self) -> Dict:
        """Return a parameter point that maximizes the uncertainty of the surrogate model.

        Returns:
            dict: A dict stored parameter to evaluate. Like:
                {"param1": value1, "param2": value2}
        """
        assert self._surrogate_model is not None
        assert self._surrogate_model.kernel_ is not None
        assert self._surrogate_model.X_train_ is not None

        # def negtive_variance_func(x: NDArray) -> float:
        #     return -self._variance_func(x)

        # result = direct(
        #     negtive_variance_func,
        #     self._space.bounds.tolist(),
        # )

        # if not result.success:
        #     raise ValueError(f"Optimization failed: {result.message}")

        # return self._space.array_to_params(result.x)

        def variance_func(x: NDArray, gp, y_max) -> NDArray:
            return self._variance_func(x)

        param_array = acq_max(
            ac=variance_func,
            gp=self._surrogate_model,
            y_max=self._space.max()["target"],
            bounds=self._space.bounds,
            random_state=self._random_state,
        )

        return self._space.array_to_params(param_array)

    def _boundary_refinement(self) -> Dict:
        """Return a parameter point that used to tighten the boundary of the failure region.

        Returns:
            dict: A dict stored parameter to evaluate. Like:
                {"param1": value1, "param2": value2}
        """

        def boundary_refinement_objective(x: NDArray, gp, y_max) -> NDArray:
            logistic_val = logistic(self._mean_func(x))
            ret = logistic_val * (1 - logistic_val) + self._lamda * self._std_deviation(
                x
            )
            print(
                f"logistic_val: {logistic_val} with shape {logistic_val.shape}\n ret: {ret} with shape {ret.shape}"
            )
            return ret

        # Try to use gradient-based optimization
        param_array = acq_max(
            ac=boundary_refinement_objective,
            gp=self._surrogate_model,
            y_max=self._space.max()["target"],
            bounds=self._space.bounds,
            random_state=self._random_state,
        )

        return self._space.array_to_params(param_array)

    def _failure_region_sampling(self) -> Dict:
        """Return a parameter point that sampling in the approximate failure region.
        Sampling from the approximate failure distribution defined by the surrogate helps
        to refine likely failure regions to ensure a better estimate of the probability of failure.

        Returns:
            Dict: A dict stored parameter to evaluate. Like:
                {"param1": value1, "param2": value2}
        """
        return {}


# def black_box(config):  # ①
#     train_loader, test_loader = load_data()  # Load some data
#     model = ConvNet().to("cpu")  # Create a PyTorch conv net
#     optimizer = torch.optim.SGD(  # Tune the optimizer
#         model.parameters(), lr=config["lr"], momentum=config["momentum"]
#     )

#     while True:
#         train_epoch(model, optimizer, train_loader)  # Train the model
#         acc = test(model, test_loader)  # Compute test accuracy
#         train.report({"mean_accuracy": acc})  # Report to Tune


# search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
# algo = OptunaSearch()  # ②

# tuner = tune.Tuner(  # ③
#     black_box,
#     tune_config=tune.TuneConfig(
#         metric="mean_accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     run_config=train.RunConfig(
#         stop={"training_iteration": 5},
#     ),
#     param_space=search_space,
# )
# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)
