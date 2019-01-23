from collections import defaultdict
from datetime import datetime
import itertools
import numpy as np
import os
import pickle
import sys
import time

sys.path.append(os.getcwd() + '/simulators')
from lbp.factor_graph import FactorGraph
from lbp.factors import Factor
from epidemics.WestAfrica import WestAfrica

from Observe import get_ebola_observation, region_observation_probability

# function to map a node in the graphical model to a numerical index
def node2index(dims, idx, time):
    return time*np.prod(dims) + idx


class LBP:
    """
    Implementation of loopy belief propagation for the WestAfrica simulator.
    """
    def __init__(self, simulation, observation, prior, iteration_limit=1, horizon=3):
        self.healthy = 0
        self.infected = 1
        self.immune = 2

        self.G = FactorGraph(numVar=0, numFactor=0)
        self.factorIdx = 0
        self.time = 0

        self.observation_history = []
        self.prior = prior

        self.add_new_layer(simulation, observation, self.prior)
        self.observation_history += [observation]

        self.iteration_limit = iteration_limit
        self.horizon = horizon

    def add_new_layer(self, simulation, observation, prior=None):
        """
        Adds a new time layer to the graph corresponding to the new measurement y. If prior is given (numpy array) and
        self.time = 0 then the layer also includes factors corresponding to the priors.

        First, this function builds the nodes for each new Region element at the current time. Then, it adds factors for
        measurements. If self.time > 0 then factors are added for the dynamics.
        """
        # number of factors and variables
        num_nodes = np.prod(simulation.dims)
        if self.time == 0 and prior is None:
            node_factors = 0
        else:
            node_factors = np.prod(simulation.dims)
        obs_factors = np.prod(simulation.dims)  # factors for observations
        factor_count = node_factors + obs_factors

        self.G.varName += [[] for _ in range(num_nodes)]
        self.G.domain += [element.state_space for element in simulation.group.values()]
        self.G.varToFactor += [[] for _ in range(num_nodes)]
        self.G.factorToVar += [[] for _ in range(factor_count)]

        # add variables and factors for each Region element
        for key, element in simulation.group.items():
            # create variable name and index
            var_idx = node2index(simulation.dims, element.numeric_id, self.time)
            xname = 'x' + str(element.numeric_id) + str(self.time)

            self.G.var.append(var_idx)
            self.G.varName[var_idx] = xname

            # scope and cardinality for measurement factor
            scope = [var_idx]
            card = [len(element.state_space)]
            val = np.zeros(card)

            # create factor for measurement
            for x in element.state_space:
                val[x] = region_observation_probability(x, observation[key])

            name = 'g_%i%i' % (element.numeric_id, self.time)
            self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

            # varToFactor[i] is list of indices of factors corresponding to variable i
            # factorToVar[j] is list of variables connected to factor j
            self.G.varToFactor[var_idx] += [self.factorIdx]
            self.G.factorToVar[self.factorIdx] += [var_idx]
            self.factorIdx += 1

            if self.time == 0 and prior is not None:
                # create prior factors
                scope = [var_idx]
                card = [len(element.state_space)]
                val = prior[key]

                name = 'b_%i%i' % (element.numeric_id, self.time)
                self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                self.G.varToFactor[var_idx] += [self.factorIdx]
                self.G.factorToVar[self.factorIdx] += [var_idx]
                self.factorIdx += 1

            # for this case, include factors between time slices
            elif self.time > 0:
                # create factor between t-1 and t
                scope = [node2index(simulation.dims, element.numeric_id, self.time-1)]
                for n in element.neighbors:
                    scope += [node2index(simulation.dims, simulation.group[n].numeric_id, self.time-1)]
                scope += [var_idx]

                # use dynamics for factor
                card = [len(element.state_space) for _ in range(len(scope))]
                val = np.zeros(card)
                iterate = [element.state_space for _ in range(len(scope))]
                for combo in itertools.product(*iterate):
                    x_tm1 = combo[0]
                    f = combo[1:-1].count(self.infected)
                    x_t = combo[-1]

                    val[combo] = element.dynamics((x_tm1, f, x_t), (0, 0))

                name = 'f_%i%i' % (element.numeric_id, self.time-1)
                self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                for idx in scope:
                    # connect this new factor to each variable's list
                    self.G.varToFactor[idx] += [self.factorIdx]

                # connect this variable to factor's list
                self.G.factorToVar[self.factorIdx] += scope
                self.factorIdx += 1

        self.time += 1

    def filter(self, simulation, observation):
        """
        Method to build the graphical model and generate the belief.
        """
        tic = time.clock()
        self.observation_history += [observation]

        # case where graphical model has not grown too large
        if 1 <= self.time < self.horizon:
            # add layer for measurement
            self.add_new_layer(simulation, observation)
            status = self.G.runParallelLoopyBP(self.iteration_limit)

        # graphical model is too large, rebuild
        else:
            # get prior at time 1
            prior = self.query_belief(simulation, 1)

            obs_hist = self.observation_history[-self.horizon:]

            # create new graph
            self.G = FactorGraph(numVar=0, numFactor=0)
            self.factorIdx = 0
            self.time = 0

            # create first layer with measurement and priors
            self.add_new_layer(simulation, obs_hist[0], prior)

            # create remaining layers with measurements
            for i in range(1, self.horizon):
                self.add_new_layer(simulation, obs_hist[i])

            status = self.G.runParallelLoopyBP(self.iteration_limit)

        toc = time.clock()
        return self.query_belief(simulation, self.time-1), status, toc-tic

    def query_belief(self, simulation, time):
        """
        Method to get belief from graphical model. Returns a dictionary where each key is the name of a Region and
        values correspond to a list with a probability for each Region state.
        """
        belief = {}

        for key in simulation.group.keys():
            var_idx = node2index(simulation.dims, simulation.group[key].numeric_id, time)
            b = self.G.estimateMarginalProbability(var_idx)
            belief[key] = b

        return belief


def run_simulation(sim_obj, iteration_limit, horizon):
    """
    Function to run a single simulation of the WestAfrica simulator using loopy belief propagation.
    """
    num_regions = np.prod(sim_obj.dims)
    control = defaultdict(lambda: (0, 0))  # no control

    # initial belief - exact state
    belief = {}
    state = sim_obj.dense_state()
    for key in state.keys():
        belief[key] = [0, 0, 0]
        belief[key][state[key]] = 1

    # instantiate filter
    robot = LBP(sim_obj, sim_obj.dense_state(), belief, iteration_limit=iteration_limit, horizon=horizon)

    observation_acc = []
    filter_acc = []
    update_time = []

    # run for only 75 iterations as process is non self-terminating
    for _ in range(75):
        # update simulator and get state
        sim_obj.update(control)
        state = sim_obj.dense_state()

        # get observation and observation accuracy
        obs = get_ebola_observation(sim_obj)
        obs_acc = [obs[name] == state[name] for name in state.keys()].count(True)/num_regions

        # run filter and get belief
        belief, status, timing = robot.filter(sim_obj, obs)
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
    For example, running 'python3 lbpWestAfrica k10' uses an iteration limit of 10.
    """

    Kmax = 1
    H = 3  # horizon parameter for LBP
    total_sims = 100

    if len(sys.argv) > 1:
        Kmax = int(sys.argv[1][1:])

    print('[LBP] Kmax = %d' % Kmax)
    print('running for %d simulation(s)' % total_sims)

    # dictionary for storing results for each simulation
    results = dict()
    results['Kmax'] = Kmax
    results['horizon'] = H
    results['total_sims'] = total_sims

    # load model information from file
    handle = open('simulators/west_africa_graph.pkl', 'rb')
    graph = pickle.load(handle)
    handle.close()

    # set initial condition
    outbreak = {('guinea', 'gueckedou'): 1}
    sim = WestAfrica(graph, outbreak)

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] start' % st)

    t0 = time.clock()
    for s in range(total_sims):
        # set random seed and reset simulation
        seed = 1000 + s
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        # run filter and store accuracies
        observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, H)
        results[seed] = dict()
        results[seed]['observation_accuracy'] = observation_accuracy
        results[seed]['LBP_accuracy'] = filter_accuracy
        results[seed]['time_per_update'] = time_data

        # periodically write to file
        if (s + 1) % 10 == 0 and (s + 1) != total_sims:
            st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            print('[%s] finished %d simulations' % (st, s + 1))

            filename = '[SAVE] ' + 'lbp_wa' + \
                       '_Kmax' + str(Kmax) + '_h' + str(H) + \
                       '_s' + str(s + 1) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()

    # save full results to file
    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] finish' % st)
    t1 = time.clock()
    print('%0.2fs = %0.2fm = %0.2fh elapsed' % (t1 - t0, (t1 - t0) / 60, (t1 - t0) / (60 * 60)))

    filename = 'lbp_wa' + \
               '_Kmax' + str(Kmax) + '_h' + str(H) + \
               '_s' + str(total_sims) + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(results, output)
    output.close()


if __name__ == '__main__':
    # the following code will run one simulation with LBP and print some statistics
    Kmax = 1
    H = 3

    print('[LBP] Kmax = %d' % Kmax)

    # load model information
    handle = open('simulators/west_africa_graph.pkl', 'rb')
    graph = pickle.load(handle)
    handle.close()

    # set initial condition
    outbreak = {('guinea', 'gueckedou'): 1}
    sim = WestAfrica(graph, outbreak)

    observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, H)
    print('median observation accuracy: %0.2f' % (np.median(observation_accuracy)*100))
    print('median filter accuracy: %0.2f' % (np.median(filter_accuracy)*100))
    print('average update time: %0.4fs' % (np.mean(time_data)))

    # the following function will run the filter for many simulations and write the results to file
    # benchmark(sys.argv)
