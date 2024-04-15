"""Manages the optimization domain and holds points."""

from warnings import warn

import numpy as np
from bayes_opt.target_space import TargetSpace

from .utils import ensure_rng


class SampleSpace(TargetSpace):
    """Holds the param-space coordinates (X) and evaluation values (Y).

    Allows for constant-time appends.

    Args:
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator


    Usage:
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = SampleSpace(target_func, pbounds, random_state=0)
    >>> x = np.array([4 , 5])
    >>> y = target_func(x)
    >>> space.register(x, y)
    >>> assert self.max()['target'] == 9
    >>> assert self.max()['params'] == {'p1': 1.0, 'p2': 2.0}
    """

    def __init__(
        self,
        pbounds,
        random_state=None,
        allow_duplicate_points=False,
    ):
        super().__init__(
            None,
            pbounds,
            random_state=random_state,
            allow_duplicate_points=allow_duplicate_points,
        )

    def probe(self, params):
        raise NotImplementedError(
            "SampleSpace can not do this operation, because it is independent with target function. "
            "Maybe you want to use TargetSpace instead."
        )

    def random_sample(self, batch=1):
        """
        Sample a batch random point uniformly from within the bounds of the space.

        Returns:
            ndarray: [batch x dim] array with dimensions corresponding to `self._keys`

        Usage:
            >>> target_func = lambda p1, p2: p1 + p2
            >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
            >>> space = TargetSpace(target_func, pbounds, random_state=0)
            >>> space.random_sample()
            array([[ 55.33253689,   0.54488318]])
        """
        data = np.empty((batch, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=batch)
        return data
