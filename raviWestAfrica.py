from collections import defaultdict
import copy
from datetime import datetime
import numpy as np
import os
import pickle
import sys
import time

sys.path.append(os.getcwd() + '/simulators')

from epidemics.WestAfrica import WestAfrica
from Observe import get_ebola_observation, region_observation_probability


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
    Class implementation of the RAVI method for the WestAfrica simulator
    """
    def __init__(self, prior, iteration_limit=1, epsilon=1e-10):
        self.healthy = 0
        self.infected = 1
        self.immune = 2

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
        posterior = copy.copy(self.prior)

        # initialize of message for each element
        for key in simulation.group.keys():
            graph[key] = dict()
            graph[key]['message'] = self.prior[key]

        # perform message-passing to update posterior factors
        status = ['Cutoff', self.iteration_limit]
        for iteration in range(self.iteration_limit):
            num_changed = 0
            for key, element in simulation.group.items():
                num_neighbors = len(element.neighbors)

                # make a CAF for neighbors' messages, a function of the number of "active" neighbors
                # for WestAfrica, active means a neighbor Region is infected
                graph[key]['CAF'] = np.zeros(num_neighbors+1)
                for l in range(2**num_neighbors):
                    xj = np.base_repr(l, base=2).zfill(num_neighbors)
                    active = xj.count('1')

                    values = []
                    for n in range(num_neighbors):
                        neighbor_key = element.neighbors[n]
                        prob = None
                        if int(xj[n]) == 0:
                            prob = graph[neighbor_key]['message'][self.healthy] + \
                                   graph[neighbor_key]['message'][self.immune]
                        elif int(xj[n]) == 1:
                            prob = graph[neighbor_key]['message'][self.infected]

                        values.append(prob)

                    graph[key]['CAF'][active] += multiply_probabilities(values)

                # construct candidate message, a function of x_{i}^{t-1} and x_{i}^{t}
                graph[key]['candidate'] = np.zeros((len(element.state_space), len(element.state_space)))
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        for active in range(num_neighbors+1):
                            values = [element.dynamics((s_tm1, active, s_t), (0, 0)), graph[key]['CAF'][active]]
                            graph[key]['candidate'][s_tm1, s_t] += multiply_probabilities(values)

                        values = [region_observation_probability(s_t, observation[key]),
                                  graph[key]['candidate'][s_tm1, s_t]]
                        graph[key]['candidate'][s_tm1, s_t] = multiply_probabilities(values)

                # update posterior factor
                q = np.zeros(len(element.state_space))
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        values = [self.prior[key][s_tm1], graph[key]['candidate'][s_tm1, s_t]]
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
                if np.argmax(q) != np.argmax(posterior[key]):
                    num_changed += 1
                posterior[key] = q

                # calculate next message
                graph[key]['next_message'] = np.zeros(len(element.state_space))
                for s_tm1 in element.state_space:
                    for s_t in element.state_space:
                        values = [q[s_t], graph[key]['candidate'][s_tm1, s_t]]
                        graph[key]['next_message'][s_tm1] += multiply_probabilities(values)

                    values = [self.prior[key][s_tm1],
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
            for key in simulation.group.keys():
                graph[key]['message'] = graph[key]['next_message']

        toc = time.clock()
        self.prior = posterior
        return posterior, status, toc-tic


def run_simulation(sim_object, iteration_limit, epsilon):
    """
    Function to run a single epidemic simulation and return accuracy metrics.
    """
    num_regions = np.prod(sim_object.dims)
    control = defaultdict(lambda: (0, 0))  # no control

    # initial belief - exact state
    belief = {}
    state = sim_object.dense_state()
    for key in state.keys():
        belief[key] = [0, 0, 0]
        belief[key][state[key]] = 1

    # instantiate filter
    robot = RAVI(belief, iteration_limit=iteration_limit, epsilon=epsilon)

    observation_acc = []
    filter_acc = []
    update_time = []

    # run for only 75 iterations as process does not self-terminate
    for _ in range(75):
        # update simulation and create dense state
        sim_object.update(control)
        state = sim_object.dense_state()

        # get observation and calculate observation accuracy
        obs = get_ebola_observation(sim_object)
        obs_acc = [obs[name] == state[name] for name in state.keys()].count(True)/num_regions

        # run filter and get belief
        belief, status, timing = robot.filter(sim_object, obs)
        estimate = {name: np.argmax(belief[name]) for name in state.keys()}
        f_acc = [estimate[name] == state[name] for name in state.keys()].count(True)/num_regions
        update_time.append(timing)

        # store accuracy for this time step
        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    return observation_acc, filter_acc, update_time


def benchmark(arguments):
    """
    Function to run many simulations and save results to file.
    The iteration limit, Kmax, can be specified on the command line, using the syntax 'ki' where i is an integer.
    For example, running 'python3 raviWestAfrica k10' uses an iteration limit of 10.
    """

    Kmax = 1
    epsilon = 1e-10
    total_sims = 100

    # if command line arguments are provided, use it
    if len(sys.argv) > 1:
        Kmax = int(sys.argv[1][1:])

    print('[RAVI] Kmax = %d' % Kmax)
    print('running for %d simulation(s)' % total_sims)

    # dictionary for storing results for each simulation
    results = dict()
    results['Kmax'] = Kmax
    results['epsilon'] = epsilon
    results['total_sims'] = total_sims

    # load model information from file
    handle = open('simulators/west_africa_graph.pkl', 'rb')
    graph = pickle.load(handle)
    handle.close()

    # initial condition
    outbreak = {('guinea', 'gueckedou'): 1}
    sim = WestAfrica(graph, outbreak)

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] start' % st)

    t0 = time.clock()
    for s in range(total_sims):
        # set random seed and reset simulation object
        seed = 1000 + s
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        # run simulation to get time history of accuracies
        observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, epsilon)
        results[seed] = dict()
        results[seed]['observation_accuracy'] = observation_accuracy
        results[seed]['RAVI_accuracy'] = filter_accuracy
        results[seed]['time_per_update'] = time_data

        # write results to file every 10 simulations
        if (s + 1) % 10 == 0 and (s + 1) != total_sims:
            st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            print('[%s] finished %d simulations' % (st, s + 1))

            filename = '[SAVE] ' + 'ravi_wa' + \
                       '_Kmax' + str(Kmax) + '_eps' + str(epsilon) + \
                       '_s' + str(s + 1) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()

    # write all results to file
    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] finish' % st)
    t1 = time.clock()
    print('%0.2fs = %0.2fm = %0.2fh elapsed' % (t1 - t0, (t1 - t0) / 60, (t1 - t0) / (60 * 60)))

    filename = 'ravi_wa' + \
               '_Kmax' + str(Kmax) + '_eps' + str(epsilon) + \
               '_s' + str(total_sims) + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(results, output)
    output.close()


if __name__ == '__main__':
    # the following code will run one simulation with RAVI and print some statistics
    Kmax = 1
    epsilon = 1e-10

    print('[RAVI] Kmax = %d' % Kmax)

    # load model information
    handle = open('simulators/west_africa_graph.pkl', 'rb')
    graph = pickle.load(handle)
    handle.close()

    # set initial condition
    outbreak = {('guinea', 'gueckedou'): 1}
    sim = WestAfrica(graph, outbreak)

    observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, epsilon)
    print('median observation accuracy: %0.2f' % (np.median(observation_accuracy)*100))
    print('median filter accuracy: %0.2f' % (np.median(filter_accuracy)*100))
    print('average update time: %0.4fs' % (np.mean(time_data)))

    # the following function will run the filter for many simulations and write the results to file
    # benchmark(sys.argv)
