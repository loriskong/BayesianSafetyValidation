import pytest

from bayesian_safety_validation import BayesianSafetyValidation


def test_bsv():
    def black_box_func(params) -> float:
        return float(
            (
                (params["x1"] + 2 * params["x2"] - 7) ** 2
                + (2 * params["x1"] + params["x2"] - 5) ** 2
            )
            <= 200
        )

    bsv = BayesianSafetyValidation(param_space={"x1": (-10, 5), "x2": (-10, 5)})

    for i in range(10):
        suggestions = bsv.suggest()
        evaluations = [black_box_func(suggestion) for suggestion in suggestions]
        print(f"suggestions: {suggestions}, evaluations: {evaluations}")
        bsv.refit(suggestions, evaluations)

    bsv.falsification()
