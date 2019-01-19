import numpy as np


def get_forest_observation(sim):
    observation = np.zeros(sim.dims).astype(int)
    for element in sim.forest.values():
        probs = [tree_observation_probability(s, element.state) for s in element.state_space]
        observation[element.position[0], element.position[1]] = np.random.choice(element.state_space, p=probs)

    return observation


def tree_observation_probability(state, observation):
    measure_correct = 0.9
    measure_wrong = 0.5*(1-measure_correct)
    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct


def get_ebola_observation(sim):
    pass


def region_observation_probability(state, observation):
    pass
