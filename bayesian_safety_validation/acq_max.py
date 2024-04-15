"""General acquisition function maxize algo.
   The acq_max implementation in bayes_opt 1.4.3 is not good enough,
   so I copied the implementation from commit 383cb29.
"""
import json
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def acq_max(
    ac,
    gp,
    y_max,
    bounds,
    random_state,
    constraint=None,
    n_warmup=10000,
    n_iter=10,
    y_max_params=None,
):
    """Find the maximum of the acquisition function.

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (10) random starting points.

    Parameters
    ----------
    ac : callable
        Acquisition function to use. Should accept an array of parameters `x`,
        an from sklearn.gaussian_process.GaussianProcessRegressor `gp` and the
        best current value `y_max` as parameters.

    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A gaussian process regressor modelling the target function based on
        previous observations.

    y_max : number
        Highest found value of the target function.

    bounds : np.ndarray
        Bounds of the search space. For `N` parameters this has shape
        `(N, 2)` with `[i, 0]` the lower bound of parameter `i` and
        `[i, 1]` the upper bound.

    random_state : np.random.RandomState
        A random state to sample from.

    constraint : ConstraintModel or None, default=None
        If provided, the acquisition function will be adjusted according
        to the probability of fulfilling the constraint.

    n_warmup : int, default=10000
        Number of points to sample from the acquisition function as seeds
        before looking for a minimum.

    n_iter : int, default=10
        Points to run L-BFGS-B optimization from.

    y_max_params : np.array
        Function parameters that produced the maximum known value given by `y_max`.

    :param y_max_params:
        Function parameters that produced the maximum known value given by `y_max`.

    Returns
    -------
    Parameters maximizing the acquisition function.

    """
    # We need to adjust the acquisition function to deal with constraints when there is some
    if constraint is not None:

        def adjusted_ac(x):
            """Acquisition function adjusted to fulfill the constraint when necessary.

            Parameters
            ----------
            x : np.ndarray
                Parameter at which to sample.


            Returns
            -------
            The value of the acquisition function adjusted for constraints.
            """
            # Transforms the problem in a minimization problem, this is necessary
            # because the solver we are using later on is a minimizer
            values = -ac(x.reshape(-1, bounds.shape[0]), gp=gp, y_max=y_max)
            p_constraints = constraint.predict(x.reshape(-1, bounds.shape[0]))

            # Slower fallback for the case where any values are negative
            if np.any(values > 0):
                # TODO: This is not exactly how Gardner et al do it.
                # Their way would require the result of the acquisition function
                # to be strictly positive, which is not the case here. For a
                # positive target value, we use Gardner's version. If the target
                # is negative, we instead slightly rescale the target depending
                # on the probability estimate to fulfill the constraint.
                return np.array(
                    [
                        value / (0.5 + 0.5 * p) if value > 0 else value * p
                        for value, p in zip(values, p_constraints)
                    ]
                )

            # Faster, vectorized version of Gardner et al's method
            return values * p_constraints

    else:
        # Transforms the problem in a minimization problem, this is necessary
        # because the solver we are using later on is a minimizer
        adjusted_ac = lambda x: -ac(x.reshape(-1, bounds.shape[0]), gp=gp, y_max=y_max)

    # Warm up with random points
    x_tries = random_state.uniform(
        bounds[:, 0], bounds[:, 1], size=(n_warmup, bounds.shape[0])
    )
    ys = -adjusted_ac(x_tries)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more thoroughly
    x_seeds = random_state.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(1 + n_iter + int(not y_max_params is None), bounds.shape[0]),
    )
    # Add the best candidate from the random sampling to the seeds so that the
    # optimization algorithm can try to walk up to that particular local maxima
    x_seeds[0] = x_max
    if not y_max_params is None:
        # Add the provided best sample to the seeds so that the optimization
        # algorithm is aware of it and will attempt to find its local maxima
        x_seeds[1] = y_max_params

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(adjusted_ac, x_try, bounds=bounds, method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -np.squeeze(res.fun) >= max_acq:
            x_max = res.x
            max_acq = -np.squeeze(res.fun)

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
