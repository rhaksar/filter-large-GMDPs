from collections import defaultdict
import numpy as np
import os
import sys

sys.path.append(os.getcwd() + '/simulators')

from fires.LatticeForest import LatticeForest
from Observe import get_forest_observation, tree_observation_probability


# linear approximation to the logarithm function
# over the interval [epsilon, 1]
def linear_approx(input, epsilon=1e-10):
    if input < epsilon:
        raise Exception('Invalid input encountered in logarithm approximation.')

    return (np.log(epsilon)/(1-epsilon))*(1-input)


class RAVI(object):
    def __init__(self, dimensions, prior, iteration_limit=1, epsilon=1e-10):
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2

        self.dimensions = dimensions
        self.prior = prior
        self.posterior = None

        self.iteration_limit = iteration_limit
        self.epsilon = epsilon

    def filter(self, simulation, observation):
        graph = dict()

        # initialize posterior as prior
        posterior = self.prior

        # initialize of message for each element
        for element in simulation.forest.values():
            key = tuple(element.position)
            graph[key] = dict()
            graph[key]['message'] = self.prior[element.position[0], element.position[1], :]

        # perform message-passing to update posterior factors
        for iteration in range(self.iteration_limit):

            for element in simulation.forest.values():
                key = tuple(element.position)
                num_neighbors = len(element.neighbors)

                # make a CAF for neighbors' messages, a function of the number of "active" neighbors
                # for the LatticeForest, active means a neighbor Tree is on_fire
                graph[key]['CAF'] = np.zeros(num_neighbors+1)
                for l in range(2**num_neighbors):
                    xj = np.base_repr(l, base=2).zfill(num_neighbors)
                    active = xj.count('1')

                    prob_product = 0
                    for n in range(num_neighbors):
                        neighbor_position = tuple(simulation.forest[element.neighbors[n]].position)
                        prob = None
                        if int(xj[n]) == 0:
                            prob = graph[neighbor_position]['message'][self.healthy] + \
                                   graph[neighbor_position]['message'][self.burnt]
                        elif int(xj[n]) == 1:
                            prob = graph[neighbor_position]['message'][self.on_fire]

                        if prob < 1e-20:
                            prob_product = -100
                            break
                        else:
                            prob_product += np.log(prob)

                    if prob_product <= -100:
                        graph[key]['CAF'][active] += 0
                    else:
                        graph[key]['CAF'][active] += np.exp(prob_product)

                # construct candidate message, a function of x_{i}^{t-1} and x_{i}^{t}
                graph[key]['candidate'] = np.zeros((len(element.state_space), len(element.state_space)))
                # first compute marginalization of dynamics times neighbors' message
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        for active in range(num_neighbors+1):
                            values = [element.dynamics((s_tm1, active, s_t), (0, 0)), graph[key]['CAF'][active]]

                            if any([v < 1e-20 for v in values]):
                                graph[key]['candidate'][s_tm1, s_t] += 0
                            else:
                                prob_product = sum([np.log(v) for v in values])
                                if prob_product <= -100:
                                    graph[key]['candidate'][s_tm1, s_t] += 0
                                else:
                                    graph[key]['candidate'][s_tm1, s_t] += np.exp(prob_product)

                        values = [tree_observation_probability(s_t, observation[element.position[0],
                                                                                element.position[1]]),
                                  graph[key]['candidate'][s_tm1, s_t]]

                        if any([v < 1e-20 for v in values]):
                            graph[key]['candidate'][s_tm1, s_t] = 0
                        else:
                            prob_product = sum([np.log(v) for v in values])
                            if prob_product <= -100:
                                graph[key]['candidate'][s_tm1, s_t] = 0
                            else:
                                graph[key]['candidate'][s_tm1, s_t] = np.exp(prob_product)

                # update posterior factor
                q = np.zeros(len(element.state_space))
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        values = [self.prior[element.position[0], element.position[1], s_tm1],
                                  graph[key]['candidate'][s_tm1, s_t]]

                        if any([v < 1e-20 for v in values]):
                            q[s_t] += 0
                        else:
                            prob_product = sum([np.log(v) for v in values])
                            if prob_product <= -100:
                                q[s_t] += 0
                            else:
                                q[s_t] += np.exp(prob_product)

                    q[s_t] = linear_approx(q[s_t], self.epsilon)

                q = [v-max(q) for v in q]
                normalization = 0
                for idx, v in enumerate(q):
                    if v <= -100:
                        q[idx] = 0
                        continue

                    q[idx] = np.exp(v)
                    normalization += q[idx]

                q /= normalization
                posterior[element.position[0], element.position[1], :] = q

                # calculate next message
                graph[key]['next_message'] = np.zeros(len(element.state_space))
                for s_tm1 in element.state_space:
                    for s_t in element.state_space:
                        values = [q[s_t], graph[key]['candidate'][s_tm1, s_t]]

                        if any([v < 1e-20 for v in values]):
                            graph[key]['next_message'][s_tm1] += 0
                        else:
                            prob_product = sum([np.log(v) for v in values])
                            if prob_product <= -100:
                                graph[key]['next_message'][s_tm1] += 0
                            else:
                                graph[key]['next_message'][s_tm1] += np.exp(prob_product)

                    values = [self.prior[element.position[0], element.position[1], s_tm1],
                              graph[key]['next_message'][s_tm1]]
                    if any([v < 1e-20 for v in values]):
                        graph[element.position]['next_message'][s_tm1] = 0
                    else:
                        prob_product = sum([np.log(v) for v in values])
                        if prob_product <= -100:
                            graph[key]['next_message'][s_tm1] = 0
                        else:
                            graph[key]['next_message'][s_tm1] = np.exp(prob_product)

                # normalize next message
                graph[key]['next_message'] /= graph[key]['next_message'].sum()

            # update messages for next round
            for element in simulation.forest.values():
                key = tuple(element.position)
                graph[key]['message'] = graph[key]['next_message']

        self.posterior = posterior
        return posterior


if __name__ == '__main__':
    dimension = 5
    control = defaultdict(lambda: (0, 0))
    seed = 1000
    np.random.seed(seed)

    sim = LatticeForest(dimension, rng=seed)

    belief = 0.33*np.ones((sim.dims[0], sim.dims[1], 3))
    belief = belief / belief.sum(axis=2, keepdims=True)

    robot = RAVI(sim.dims, belief)

    for _ in range(1):
        sim.update(control)

        obs = get_forest_observation(sim)
        print(sim.dense_state())
        print(obs)

        belief = robot.filter(sim, obs)
        estimate = np.argmax(belief, axis=2)
        print(estimate)

    print()
