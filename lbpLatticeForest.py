from collections import defaultdict
from datetime import datetime
import itertools
import numpy as np
import os
import pickle
import sys
import time

sys.path.append(os.path.dirname(os.getcwd()) + '/simulators')
from lbp import LBP
from fires.LatticeForest import LatticeForest
from Observe import get_forest_observation, tree_observation_probability


# # function to map a node in the graphical model to a numeric index
# def node2index(dims, idx, t):
#     return t*np.prod(dims) + idx
#
#
# class LBP:
#     """
#     Implementation of loopy belief propagation for the LatticeForest simulator.
#     """
#     def __init__(self, simulation, observation, prior, iteration_limit=1, horizon=3):
#         self.healthy = 0
#         self.on_fire = 1
#         self.burnt = 2
#
#         self.G = FactorGraph(numVar=0, numFactor=0)
#         self.factorIdx = 0
#         self.time = 0
#         self.observation_history = []
#         self.prior = prior
#
#         self.add_new_layer(simulation, observation, self.prior)
#         self.observation_history += [observation]
#
#         self.iteration_limit = iteration_limit
#         self.horizon = horizon
#
#     def add_new_layer(self, simulation, observation, prior=None):
#         """
#         Adds a new time layer to the graph corresponding to the new measurement y. If prior is given (numpy array) and
#         self.time = 0 then the layer also includes factors corresponding to the priors.
#
#         First, this function builds the nodes for each new Tree element at the current time. Then, it adds factors for
#         measurements. If self.time >0 then factors are added for the dynamics.
#         """
#         # number of factors and variables
#         num_nodes = np.prod(simulation.dims)
#         if self.time == 0 and prior is None:
#             node_factors = 0
#         else:
#             node_factors = np.prod(simulation.dims)
#
#         obs_factors = np.prod(simulation.dims)  # factors for observations
#         factor_count = node_factors + obs_factors
#
#         self.G.varName += [[] for _ in range(num_nodes)]
#         self.G.domain += [element.state_space for element in simulation.group.values()]
#         self.G.varToFactor += [[] for _ in range(num_nodes)]
#         self.G.factorToVar += [[] for _ in range(factor_count)]
#
#         # add factors for each Tree element
#         for element in simulation.group.values():
#             # create variable name and index
#             row, col = element.position
#             var_idx = node2index(simulation.dims, element.numeric_id, self.time)
#             xname = 'x' + str(row) + str(col) + str(self.time)
#
#             self.G.var.append(var_idx)
#             self.G.varName[var_idx] = xname
#
#             # scope and cardinality for measurement factor
#             scope = [var_idx]
#             card = [len(element.state_space)]
#             val = np.zeros(card)
#
#             # create factor for measurement
#             for x in element.state_space:
#                 val[x] = tree_observation_probability(x, observation[row, col])
#
#             name = 'g_%i%i%i' % (row, col, self.time)
#             self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))
#
#             # varToFactor[i] is list of indices of factors corresponding to variable i
#             # factorToVar[j] is list of variables connected to factor j
#             self.G.varToFactor[var_idx] += [self.factorIdx]
#             self.G.factorToVar[self.factorIdx] += [var_idx]
#             self.factorIdx += 1
#
#             if self.time == 0 and prior is not None:
#                 # create prior factors
#                 scope = [var_idx]
#                 card = [len(element.state_space)]
#                 val = prior[row, col]
#
#                 name = 'b_%i%i%i' % (row, col, self.time)
#                 self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))
#
#                 self.G.varToFactor[var_idx] += [self.factorIdx]
#                 self.G.factorToVar[self.factorIdx] += [var_idx]
#                 self.factorIdx += 1
#
#             # for this case, include factors between time slices
#             elif self.time > 0:
#                 # create factor between t-1 and t
#                 scope = [node2index(simulation.dims, element.numeric_id, self.time-1)]
#                 for n in element.neighbors:
#                     scope += [node2index(simulation.dims, simulation.group[n].numeric_id, self.time-1)]
#                 scope += [var_idx]
#
#                 # use dynamics for factor
#                 card = [len(element.state_space) for _ in range(len(scope))]
#                 val = np.zeros(card)
#                 iterate = [element.state_space for _ in range(len(scope))]
#                 for combo in itertools.product(*iterate):
#                     x_tm1 = combo[0]
#                     f = combo[1:-1].count(self.on_fire)
#                     x_t = combo[-1]
#
#                     val[combo] = element.dynamics((x_tm1, f, x_t), (0, 0))
#
#                 name = 'f_%i%i%i' % (row, col, self.time-1)
#                 self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))
#
#                 for idx in scope:
#                     # connect this new factor to each variable's list
#                     self.G.varToFactor[idx] += [self.factorIdx]
#
#                 # connect this variable to factor's list
#                 self.G.factorToVar[self.factorIdx] += scope
#                 self.factorIdx += 1
#
#         self.time += 1
#
#     def filter(self, simulation, observation):
#         """
#         Method to build the graphical model and generate the belief.
#         """
#         tic = time.clock()
#         self.observation_history += [observation]
#
#         # case where graphical model has not grown too large
#         if 1 <= self.time < self.horizon:
#             self.add_new_layer(simulation, observation)
#             status = self.G.runParallelLoopyBP(self.iteration_limit)
#
#         # graphical model is too large, rebuild
#         else:
#             # get prior at time 1
#             prior = self.query_belief(simulation, 1)
#
#             obs_hist = self.observation_history[-self.horizon:]
#
#             # rebuild model
#             self.G = FactorGraph(numVar=0, numFactor=0)
#             self.factorIdx = 0
#             self.time = 0
#
#             # create first layer with measurement and priors
#             self.add_new_layer(simulation, obs_hist[0], prior)
#
#             # create remaining layers with measurements
#             for i in range(1, self.horizon):
#                 self.add_new_layer(simulation, obs_hist[i])
#
#             status = self.G.runParallelLoopyBP(self.iteration_limit)
#
#         toc = time.clock()
#         return self.query_belief(simulation, self.time-1), status, toc-tic
#
#     def query_belief(self, simulation, time):
#         """
#         Method to get belief from graphical model. Returns a numpy array where (row, col) corresponds to a 1D array
#         containing the state probabilities for the Tree positioned at (row, col).
#         """
#         belief = np.zeros((simulation.dims[0], simulation.dims[1], 3))
#
#         for element in simulation.group.values():
#             var_idx = node2index(simulation.dims, element.numeric_id, time)
#             row, col = element.position
#             b = self.G.estimateMarginalProbability(var_idx)
#             belief[row, col] = b
#
#         return belief

def obs_model(tree, state_value, obs_value):
    return tree_observation_probability(state_value, obs_value)


def trans_model(tree, combo):
    x_tm1 = combo[0]
    f = combo[1:-1].count(tree.on_fire)
    x_t = combo[-1]
    return tree.dynamics((x_tm1, f, x_t))


def run_simulation(sim_obj, iteration_limit, horizon):
    """
    Function to run a single simulation of the LatticeForest simulator using loopy belief propagation.
    """
    num_trees = np.prod(sim_obj.dims)

    # initial belief - exact state
    belief = np.zeros((sim_obj.dims[0], sim_obj.dims[1], 3))
    state = sim_obj.dense_state()
    p = np.where(state == 0)
    belief[p[0], p[1], :] = [1, 0, 0]
    p = np.where(state == 1)
    belief[p[0], p[1], :] = [0, 1, 0]
    p = np.where(state == 2)
    belief[p[0], p[1], :] = [0, 0, 1]
    belief = belief / belief.sum(axis=2, keepdims=True)

    # instantiate filter
    robot = LBP(sim_obj, sim_obj.dense_state(), obs_model, trans_model, belief,
                iteration_limit=iteration_limit, horizon=horizon)

    observation_acc = []
    filter_acc = []
    update_time = []

    # run until forest fire terminates
    while not sim_obj.end:
        # update simulator and get state
        sim_obj.update()
        state = sim_obj.dense_state()

        # get observation and observation accuracy
        obs = get_forest_observation(sim_obj)
        obs_acc = np.sum(obs == state)/num_trees

        # run filter and get belief
        belief, status, timing = robot.filter(sim_obj, obs, obs_model, trans_model)
        # estimate = np.argmax(belief, axis=2)
        estimate = np.zeros_like(state)
        for key in belief.keys():
            estimate[key] = np.argmax(belief[key])
        f_acc = np.sum(estimate == state)/num_trees
        update_time.append(timing)

        # store accuracy for this time step
        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    return observation_acc, filter_acc, update_time


def benchmark(arguments):
    """
    Function to run many simulations and save results to file.
    The forest size, dimension, and iteration limit, Kmax, can be specified on the command line.
    For example, running 'python3 lbpLatticeForest d3 k10' uses dimension = 3 and Kmax = 10
    """
    dimension = 3
    Kmax = 1
    H = 3  # horizon parameter
    total_sims = 1

    if len(arguments) > 1:
        dimension = int(arguments[1][1:])
        Kmax = int(arguments[2][1:])

    print('[LBP] dimension = %d, Kmax = %d' % (dimension, Kmax))
    print('running for %d simulation(s)' % total_sims)

    # dictionary for results for each simulation
    results = dict()
    results['dimension'] = dimension
    results['Kmax'] = Kmax
    results['horizon'] = H
    results['total_sims'] = total_sims

    # create a non-uniform grid of fire propagation parameters to model wind effects
    alpha = dict()
    alpha_start = 0.1
    alpha_end = 0.4
    for r in range(dimension):
        for c in range(dimension):
            alpha[(r, c)] = alpha_start + (c / (dimension - 1)) * (alpha_end - alpha_start)
    sim = LatticeForest(dimension, alpha=alpha)

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] start' % st)

    t0 = time.clock()
    for s in range(total_sims):
        # set random seed and reset simulation
        seed = 1000 + s
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()

        # run simulation and get results
        observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, H)
        results[seed] = dict()
        results[seed]['observation_accuracy'] = observation_accuracy
        results[seed]['LBP_accuracy'] = filter_accuracy
        results[seed]['time_per_update'] = time_data

        # periodically write to file
        if (s + 1) % 10 == 0 and (s + 1) != total_sims:
            st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            print('[%s] finished %d simulations' % (st, s + 1))

            filename = '[SAVE] ' + 'lbp_d' + str(dimension) + \
                       '_Kmax' + str(Kmax) + '_h' + str(H) + \
                       '_s' + str(s + 1) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] finish' % st)
    t1 = time.clock()
    print('%0.2fs = %0.2fm = %0.2fh elapsed' % (t1 - t0, (t1 - t0) / 60, (t1 - t0) / (60 * 60)))

    # save final results to file
    filename = 'lbp_d' + str(dimension) + \
               '_Kmax' + str(Kmax) + '_h' + str(H) + \
               '_s' + str(total_sims) + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(results, output)
    output.close()


if __name__ == '__main__':
    # the following code will run one simulation and print some statistics
    dimension = 3
    Kmax = 1
    H = 3

    print('[LBP] dimension = %d, Kmax = %d' % (dimension, Kmax))

    # create non-uniform grid of fire propagation parameters to model wind effects
    alpha = dict()
    alpha_start = 0.1
    alpha_end = 0.4
    for row in range(dimension):
        for col in range(dimension):
            alpha[(row, col)] = alpha_start + (col/(dimension-1))*(alpha_end-alpha_start)
    sim = LatticeForest(dimension, alpha=alpha)

    observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, H)
    print('median observation accuracy: %0.2f' % (np.median(observation_accuracy)*100))
    print('median filter accuracy: %0.2f' % (np.median(filter_accuracy)*100))
    print('average update time: %0.4fs' % (np.mean(time_data)))

    # this function will run many simulations and save results to file
    # benchmark(sys.argv)
