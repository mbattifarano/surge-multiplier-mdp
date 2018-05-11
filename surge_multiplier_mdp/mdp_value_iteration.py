import itertools
from collections import namedtuple
from toolz import curry
import numpy as np

Parameters = namedtuple('Parameters', ['alpha', 'beta1', 'beta0',
                                       'p_stay', 'demand_rate', 'base_cost', 'empty_cost',
                                       'discount'])
State = namedtuple('State', ['users', 'vehicles'])


@curry
def vehicle_in_rate(beta0, beta1, multiplier):
    """Arrival rate of vehicles given a surge multiplier"""
    return beta1 * np.log(multiplier) + beta0


@curry
def acceptance_probability(alpha, multiplier):
    """The probability of a user accepting a ride given the multiplier"""
    return np.exp(-alpha * (multiplier - 1))


@curry
def vehicles_in(rate, shape):
    """Sample the number of vehicles entering the location"""
    return np.random.poisson(rate, shape)


@curry
def users_matched(accept_probability, users, vehicles, shape):
    """Sample the number of users matched to vehicles"""
    samples = np.random.binomial(users, accept_probability, shape)
    return np.minimum(samples, vehicles)


@curry
def users_remaining(parameters, users, matched, shape):
    """Sample the number of users remaining"""
    return np.random.binomial(users - matched, parameters.p_stay, shape)


@curry
def users_in(parameters, shape):
    return np.random.poisson(parameters.demand_rate, shape)


def transition_users(parameters, state, matched, shape):
    return (users_remaining(parameters, state.users, matched, shape)
            + users_in(parameters, shape))


def transition_vehicles(parameters, state, multiplier, matched, shape):
    rate = vehicle_in_rate(parameters.beta0, parameters.beta1, multiplier)
    return (
        state.vehicles
        - matched
        + vehicles_in(rate, shape)
    )


@curry
def transition(parameters, state, multiplier, shape):
    matched = users_matched(acceptance_probability(parameters.alpha, multiplier),
                            state.users, state.vehicles, shape)
    return (
        matched,
        transition_users(parameters, state, matched, shape),
        transition_vehicles(parameters, state, multiplier, matched, shape)
    )


def reward(parameters, state, matched, multiplier):
    return (
        matched * multiplier * parameters.base_cost
        - (state.vehicles - matched) * parameters.empty_cost
    )


@curry
def expected_q(parameters, state, values, shape, m):
    max_v, max_u = values.shape
    (matched, u_next, v_next) = transition(parameters, state, m, shape)
    u_next = np.minimum(u_next, max_u-1)
    v_next = np.minimum(v_next, max_v-1)
    q = reward(parameters, state, matched, m) + parameters.discount * values[v_next, u_next]
    return q.mean()


def value_iteration(tolerance, initial_value, multipliers, parameters, shape, max_iter=200, debug=False):
    v_states, u_states = initial_value.shape
    values = initial_value.copy()
    policies = np.zeros([v_states, u_states, len(multipliers)])
    policy = np.zeros([v_states, u_states])
    convergence = []
    for k in range(max_iter):
        if debug:
            print("iteration: %d" % k)
        new_values = values.copy()
        for v, u in itertools.product(range(v_states), range(u_states)):
            state = State(users=u, vehicles=v)
            policies[v, u, :] = list(map(expected_q(parameters, state, values, shape), multipliers))
            idx = policies[v, u, :].squeeze().argmax()
            policy[v, u] = multipliers[idx]
            new_values[v, u] = policies[v, u, idx]
        convergence.append(np.nan_to_num(abs(new_values - values)/values).max())
        if debug:
            mape = 100.0 * convergence[k]
            print("convergence distance: %0.4f%%" % mape)
        if convergence[k] < tolerance:
            break
        values = new_values
    return policy, values, policies, np.array(convergence)
