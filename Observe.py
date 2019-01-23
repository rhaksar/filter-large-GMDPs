import numpy as np


def get_forest_observation(sim):
    """
    Generates an observation of the LatticeForest. Returns a matrix where each position (row, col) corresponds
    to a noisy state estimate of the Tree location at (row, col).
    """
    observation = np.zeros(sim.dims).astype(int)
    for element in sim.group.values():
        probs = [tree_observation_probability(s, element.state) for s in element.state_space]
        observation[element.position[0], element.position[1]] = np.random.choice(element.state_space, p=probs)

    return observation


def tree_observation_probability(state, observation):
    """
    Measurement model for a single Tree element for the LatticeForest.
    Returns a probability of the combination (state, observation).
    """
    measure_correct = 0.9
    measure_wrong = 0.5*(1-measure_correct)
    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct


def get_ebola_observation(sim):
    """
    Generates an observation of the WestAfrica simulator. Returns a dictionary where each key is a Region name and
    the value is a noisy state estimate of the Region.
    """
    observation = {}
    for name in sim.group.keys():
        probs = [region_observation_probability(s, sim.group[name].state) for s in sim.group[name].state_space]
        observation[name] = np.random.choice(sim.group[name].state_space, p=probs)

    return observation


def region_observation_probability(state, observation):
    """
    Measurement model for a single REgion element for the WestAfrica simulator.
    Returns a probability of the combination (state, observation).
    """
    measure_correct = 0.85
    measure_wrong = 0.5*(1-measure_correct)
    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct
