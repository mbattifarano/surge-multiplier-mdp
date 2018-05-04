from surge_multiplier_mdp import mdp_value_iteration
import numpy as np


def test_value_iteration():
    iterations = 3
    initial_value = np.ones([10, 5])
    multipliers = np.array([1, 2, 3])
    p = mdp_value_iteration.Parameters(2, 0.5, 0.01, 0.95, 2, 5, 2, 0.9)
    shape = 10
    policy, values, policies, convergence = mdp_value_iteration.value_iteration(
        iterations,
        initial_value,
        multipliers,
        p,
        shape
    )
    assert True
