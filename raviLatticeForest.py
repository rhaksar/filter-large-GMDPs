from collections import defaultdict
import numpy as np
import os
import sys

sys.path.append(os.getcwd() + '/simulators')

from fires.LatticeForest import LatticeForest
from Observe import get_forest_observation, tree_observation_probability


# helper function to carefully multiply probabilities
def multiply_probabilities(values):
    if any([v < 1e-20 for v in values]):
        return 0
    else:
        sum_log = sum([np.log(v) for v in values])
        if sum_log <= -100:
            return 0
        else:
            return np.exp(sum_log)


# linear approximation to the logarithm function
# over the interval [epsilon, 1]
def linear_approx(x, epsilon=1e-10):
    if x < epsilon:
        x = epsilon

    return (np.log(epsilon)/(1-epsilon))*(1-x)


class RAVI(object):
    def __init__(self, prior, iteration_limit=1, epsilon=1e-10):
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2

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
        status = 'iteration limit reached'
        for iteration in range(self.iteration_limit):
            num_converged = 0
            for element in simulation.forest.values():
                key = tuple(element.position)
                num_neighbors = len(element.neighbors)

                # make a CAF for neighbors' messages, a function of the number of "active" neighbors
                # for the LatticeForest, active means a neighbor Tree is on_fire
                graph[key]['CAF'] = np.zeros(num_neighbors+1)
                for l in range(2**num_neighbors):
                    xj = np.base_repr(l, base=2).zfill(num_neighbors)
                    active = xj.count('1')

                    values = []
                    for n in range(num_neighbors):
                        neighbor_position = tuple(simulation.forest[element.neighbors[n]].position)
                        prob = None
                        if int(xj[n]) == 0:
                            prob = graph[neighbor_position]['message'][self.healthy] + \
                                   graph[neighbor_position]['message'][self.burnt]
                        elif int(xj[n]) == 1:
                            prob = graph[neighbor_position]['message'][self.on_fire]

                        values.append(prob)

                    graph[key]['CAF'][active] += multiply_probabilities(values)

                # construct candidate message, a function of x_{i}^{t-1} and x_{i}^{t}
                graph[key]['candidate'] = np.zeros((len(element.state_space), len(element.state_space)))
                # first compute marginalization of dynamics times neighbors' message
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        for active in range(num_neighbors+1):
                            values = [element.dynamics((s_tm1, active, s_t), (0, 0)), graph[key]['CAF'][active]]
                            graph[key]['candidate'][s_tm1, s_t] += multiply_probabilities(values)

                        values = [tree_observation_probability(s_t, observation[element.position[0],
                                                                                element.position[1]]),
                                  graph[key]['candidate'][s_tm1, s_t]]
                        graph[key]['candidate'][s_tm1, s_t] = multiply_probabilities(values)

                # update posterior factor
                q = np.zeros(len(element.state_space))
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        values = [self.prior[element.position[0], element.position[1], s_tm1],
                                  graph[key]['candidate'][s_tm1, s_t]]
                        q[s_t] += multiply_probabilities(values)

                    q[s_t] = linear_approx(q[s_t], self.epsilon)

                # normalize posterior factor
                q = [v-max(q) for v in q]
                normalization = 0
                for idx, v in enumerate(q):
                    if v <= -100:
                        q[idx] = 0
                        continue

                    q[idx] = np.exp(v)
                    normalization += q[idx]

                q /= normalization
                if np.argmax(q) == np.argmax(posterior[element.position[0], element.position[1], :]):
                    num_converged += 1
                posterior[element.position[0], element.position[1], :] = q

                # calculate next message
                graph[key]['next_message'] = np.zeros(len(element.state_space))
                for s_tm1 in element.state_space:
                    for s_t in element.state_space:
                        values = [q[s_t], graph[key]['candidate'][s_tm1, s_t]]
                        graph[key]['next_message'][s_tm1] += multiply_probabilities(values)

                    values = [self.prior[element.position[0], element.position[1], s_tm1],
                              graph[key]['next_message'][s_tm1]]
                    graph[key]['next_message'][s_tm1] = multiply_probabilities(values)

                # normalize next message
                graph[key]['next_message'] /= graph[key]['next_message'].sum()

            # if less than 1% of nodes are changing their max likelihood estimate, break
            if num_converged/np.prod(simulation.dims) <= 0.01:
                status = 'converged'
                break

            # update messages for next round
            for element in simulation.forest.values():
                key = tuple(element.position)
                graph[key]['message'] = graph[key]['next_message']

        self.posterior = posterior
        return posterior, status


if __name__ == '__main__':
    dimension = 25
    control = defaultdict(lambda: (0, 0))
    seed = 1000
    np.random.seed(seed)

    sim = LatticeForest(dimension, rng=seed)

    belief = np.zeros((sim.dims[0], sim.dims[1], 3))
    state = sim.dense_state()
    p = np.where(state == 0)
    belief[p[0], p[1], :] = [1, 0, 0]
    p = np.where(state == 1)
    belief[p[0], p[1], :] = [0, 1, 0]
    p = np.where(state == 2)
    belief[p[0], p[1], :] = [0, 0, 1]
    belief = belief / belief.sum(axis=2, keepdims=True)

    robot = RAVI(belief, iteration_limit=10, epsilon=1e-10)

    observation_errors = []
    filter_errors = []
    # for _ in range(5):
    while not sim.end:
        sim.update(control)
        state = sim.dense_state()

        obs = get_forest_observation(sim)
        obs_err = np.sum(obs != state)
        # print(sim.dense_state())
        # print(obs)

        belief, _ = robot.filter(sim, obs)
        estimate = np.argmax(belief, axis=2)
        filter_err = np.sum(estimate != state)
        # print(belief)
        # print(estimate)
        print('errors:', filter_err)

        filter_errors.append(filter_err)
        observation_errors.append(obs_err)

    grid_size = sim.dims[0]*sim.dims[1]
    print('observation, filter min error: %0.4f, %0.4f' % (np.amin(observation_errors) * 100 / grid_size,
                                                           np.amin(filter_errors) * 100 / grid_size))
    print('observation, filter median error: %0.4f, %0.4f' % (np.median(observation_errors)*100/grid_size,
                                                              np.median(filter_errors)*100/grid_size))
    print('observation, filter max error: %0.4f, %0.4f' % (np.amax(observation_errors) * 100 / grid_size,
                                                           np.amax(filter_errors) * 100 / grid_size))
    print('observation, filter error variance: %0.4f, %0.4f' % (np.var(observation_errors) * 100 / grid_size,
                                                                np.var(filter_errors) * 100 / grid_size))
    print()
