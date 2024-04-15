import numpy as np


def ensure_rng(random_state=None):
    """Create a random number generator based on an optional seed.

    Args:
        random_state : np.random.RandomState or int or None, default=None
            Random state to use. if `None`, will create an unseeded random state.
            If `int`, creates a state using the argument as seed. If a
            `np.random.RandomState` simply returns the argument.

    Returns:
        np.random.RandomState

    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


def logistic(x):
    """Logistic function."""
    return 1 / (1 + np.exp(-x))


def phi(x, epsilon: float = 10e-5):
    """This function is used to avoid (0,0)."""
    return x * (1 - epsilon) + (1 - x) * epsilon


def inv_phi(x, epsilon: float = 10e-5):
    """Inverse of phi."""
    return (x - epsilon) / (1 - 2 * epsilon)


def logit(x):
    """Logit function."""
    return np.log(x / (1 - x))
