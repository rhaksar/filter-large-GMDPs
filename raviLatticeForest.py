from collections import defaultdict
from datetime import datetime
import numpy as np
import os
import pickle
import sys
import time

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
    """
    Class implementation of the RAVI method for the LatticeForest simulator
    """
    def __init__(self, prior, iteration_limit=1, epsilon=1e-10):
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2

        self.prior = prior

        self.iteration_limit = iteration_limit
        self.epsilon = epsilon

    def filter(self, simulation, observation):
        """
        Produce posterior factors.
        """
        tic = time.clock()
        graph = dict()

        # initialize posterior as prior
        posterior = np.copy(self.prior)
        # posterior = 0.33*np.ones_like(self.prior)
        # posterior = posterior / posterior.sum(axis=2, keepdims=True)

        # initialize of message for each element
        for element in simulation.forest.values():
            key = tuple(element.position)
            graph[key] = dict()
            graph[key]['message'] = self.prior[element.position[0], element.position[1], :]

        # perform message-passing to update posterior factors
        status = ['Cutoff', self.iteration_limit]
        for iteration in range(self.iteration_limit):
            num_changed = 0
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
                if np.argmax(q) != np.argmax(posterior[element.position[0], element.position[1], :]):
                    num_changed += 1
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
                d = graph[key]['next_message']
                d = [v-max(d) for v in d]
                normalization = 0
                for idx, v in enumerate(d):
                    if v <= -100:
                        d[idx] = 0
                        continue

                    d[idx] = np.exp(v)
                    normalization += d[idx]
                d /= normalization
                graph[key]['next_message'] = d

            # if less than 1% of nodes are changing their max likelihood estimate, break
            if iteration > 1 and num_changed/np.prod(simulation.dims) <= 0.01:
                status = ['Converged', iteration]
                break

            # update messages for next round
            for element in simulation.forest.values():
                key = tuple(element.position)
                graph[key]['message'] = graph[key]['next_message']

        toc = time.clock()
        self.prior = posterior
        return posterior, status, toc-tic


def run_simulation(sim_object, iteration_limit, epsilon):
    num_trees = np.prod(sim_object.dims)
    control = defaultdict(lambda: (0, 0))

    # initial belief
    belief = np.zeros((sim_object.dims[0], sim_object.dims[1], 3))
    state = sim_object.dense_state()
    p = np.where(state == 0)
    belief[p[0], p[1], :] = [1, 0, 0]
    p = np.where(state == 1)
    belief[p[0], p[1], :] = [0, 1, 0]
    p = np.where(state == 2)
    belief[p[0], p[1], :] = [0, 0, 1]
    belief = belief / belief.sum(axis=2, keepdims=True)

    robot = RAVI(belief, iteration_limit=iteration_limit, epsilon=epsilon)

    observation_acc = []
    filter_acc = []
    update_time = []

    while not sim_object.end:
        sim_object.update(control)
        state = sim_object.dense_state()

        obs = get_forest_observation(sim_object)
        obs_acc = np.sum(obs == state)/num_trees

        belief, status, timing = robot.filter(sim_object, obs)
        estimate = np.argmax(belief, axis=2)
        f_acc = np.sum(estimate == state)/num_trees
        update_time.append(timing)

        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    return observation_acc, filter_acc, update_time


if __name__ == '__main__':
    dimension = 3
    Kmax = 1
    epsilon = 1e-5
    total_sims = 20

    results = dict()
    results['dimension'] = dimension
    results['Kmax'] = Kmax
    results['epsilon'] = epsilon
    results['total_sims'] = total_sims

    alpha = dict()
    alpha_start = 0.1
    alpha_end = 0.4
    for r in range(dimension):
        for c in range(dimension):
            alpha[(r, c)] = alpha_start + (c/(dimension-1))*(alpha_end - alpha_start)
    sim = LatticeForest(dimension, alpha=alpha)

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] start' % st)

    t0 = time.clock()
    for s in range(total_sims):
        seed = 1000+s
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, epsilon)
        results[seed] = dict()
        results[seed]['observation_accuracy'] = observation_accuracy
        results[seed]['RAVI_accuracy'] = filter_accuracy
        results[seed]['time_per_update'] = time_data

        if (s+1) % 10 == 0 and (s+1) != total_sims:
            st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            print('[%s] finished %d simulations' % (st, s+1))

            filename = '[SAVE] ' + 'ravi_d' + str(dimension) + \
                       '_Kmax' + str(Kmax) + '_eps' + str(epsilon) + \
                       '_s' + str(s+1) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] finish' % st)
    t1 = time.clock()
    print('%0.2fs = %0.2fm = %0.2fh elapsed' % (t1-t0, (t1-t0)/60, (t1-t0)/(60*60)))

    filename = 'ravi_d' + str(dimension) + \
               '_Kmax' + str(Kmax) + '_eps' + str(epsilon) + \
               '_s' + str(total_sims) + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(results, output)
    output.close()
