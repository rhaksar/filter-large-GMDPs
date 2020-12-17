from collections import defaultdict
from datetime import datetime
import numpy as np
import pickle
import sys
import time

from simulators.fires.LatticeForest import LatticeForest
from filters.observe import get_forest_observation, tree_observation_probability
from filters.ravi import RAVI, multiply_probabilities


def candidate_message_forest(tree, filter_graph, observation, control):
    """
    Function to build a candidate message for a Tree in the LatticeForest simulator.
    """
    candidate = np.zeros((len(tree.state_space), len(tree.state_space)))
    num_neighbors = len(tree.neighbors)

    # make a CAF for neighbors' messages, a function of the number of "active" neighbors
    # for the LatticeForest, active means a neighbor Tree is on_fire
    caf = np.zeros(num_neighbors+1)
    for l in range(2**num_neighbors):
        xj = np.base_repr(l, base=2).zfill(num_neighbors)
        active = xj.count('1')

        values = []
        for n in range(num_neighbors):
            neighbor_key = tree.neighbors[n]
            prob = None
            if int(xj[n]) == 0:
                prob = 1 - filter_graph[neighbor_key]['message'][tree.on_fire]
                # prob = filter_graph[neighbor_key]['message'][tree.healthy] + \
                #        filter_graph[neighbor_key]['message'][tree.burnt]
            elif int(xj[n]) == 1:
                prob = filter_graph[neighbor_key]['message'][tree.on_fire]

            values.append(prob)

        caf[active] += multiply_probabilities(values)

    # construct candidate message, a function of x_{i}^{t-1} and x_{i}^{t}
    for s_t in tree.state_space:
        for s_tm1 in tree.state_space:
            for active in range(num_neighbors+1):
                values = [tree.dynamics((s_tm1, active, s_t), control[tuple(tree.position)]), caf[active]]
                candidate[s_tm1, s_t] += multiply_probabilities(values)

            values = [tree_observation_probability(s_t, observation[tree.position[0], tree.position[1]]),
                      candidate[s_tm1, s_t]]
            candidate[s_tm1, s_t] = multiply_probabilities(values)

    return candidate


def run_simulation(sim_object, iteration_limit, epsilon):
    """
    Function to run a single forest fire simulation and return accuracy metrics.
    """
    num_trees = np.prod(sim_object.dims)

    # initial belief - exact state
    belief = np.zeros((sim_object.dims[0], sim_object.dims[1], 3))
    state = sim_object.dense_state()
    p = np.where(state == 0)
    belief[p[0], p[1], :] = [1, 0, 0]
    p = np.where(state == 1)
    belief[p[0], p[1], :] = [0, 1, 0]
    p = np.where(state == 2)
    belief[p[0], p[1], :] = [0, 0, 1]
    belief = belief / belief.sum(axis=2, keepdims=True)

    # instantiate filter
    robot = RAVI(belief, iteration_limit=iteration_limit, epsilon=epsilon)

    observation_acc = []
    filter_acc = []
    update_time = []

    # run until forest fire terminates
    while not sim_object.end:
        # update simulator and get state
        sim_object.update()
        state = sim_object.dense_state()

        # get observation and observation accuracy
        obs = get_forest_observation(sim_object)
        obs_acc = np.sum(obs == state)/num_trees

        # run filter and get belief
        def build_candidate(element, graph):
            return candidate_message_forest(element, graph, obs, defaultdict(lambda: (0, 0)))
        belief, status, timing = robot.filter(sim_object, build_candidate)
        estimate = np.argmax(belief, axis=2)
        f_acc = np.sum(estimate == state)/num_trees
        update_time.append(timing)

        # store results at current time step
        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    return observation_acc, filter_acc, update_time


def benchmark(arguments):
    """
    Function to run many simulations and save results to file.
    The forest size, dimension, and iteration limit, Kmax, can be specified on the command line.
    For example, running 'python3 raviLatticeForest d3 k10' uses dimension = 3 and Kmax = 10
    """
    dimension = 3
    Kmax = 1
    epsilon = 1e-10
    total_sims = 100

    # use command line arguments if provided
    if len(arguments) > 1:
        dimension = int(arguments[1][1:])
        Kmax = int(arguments[2][1:])

    print('[RAVI] dimension = %d, Kmax = %d' % (dimension, Kmax))
    print('[RAVI] epsilon = %e' % epsilon)
    print('running for %d simulation(s)' % total_sims)

    # dictionary for storing results for each simulation
    results = dict()
    results['dimension'] = dimension
    results['Kmax'] = Kmax
    results['epsilon'] = epsilon
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

        # run filter and get results
        observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, epsilon)
        results[seed] = dict()
        results[seed]['observation_accuracy'] = observation_accuracy
        results[seed]['RAVI_accuracy'] = filter_accuracy
        results[seed]['time_per_update'] = time_data

        # periodically write to file
        if (s + 1) % 10 == 0 and (s + 1) != total_sims:
            st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            print('[%s] finished %d simulations' % (st, s + 1))
        #
        #     filename = '[SAVE] ' + 'ravi_d' + str(dimension) + \
        #                '_Kmax' + str(Kmax) + '_eps' + str(epsilon) + \
        #                '_s' + str(s + 1) + '.pkl'
        #     output = open(filename, 'wb')
        #     pickle.dump(results, output)
        #     output.close()

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] finish' % st)
    t1 = time.clock()
    print('%0.2fs = %0.2fm = %0.2fh elapsed' % (t1 - t0, (t1 - t0) / 60, (t1 - t0) / (60 * 60)))

    # write full results to file
    filename = 'ravi_d' + str(dimension) + \
               '_Kmax' + str(Kmax) + '_eps' + str(epsilon) + \
               '_s' + str(total_sims) + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(results, output)
    output.close()


if __name__ == '__main__':
    # the following code will run one simulation and print some statistics
    # seed = 1023  # 1064
    # np.random.seed(seed)
    # dimension = 25
    # Kmax = 1
    # epsilon = 1e-10
    #
    # print('[RAVI] dimension = %d, Kmax = %d' % (dimension, Kmax))
    #
    # # create non-uniform grid of fire propagation parameters to model wind effects
    # alpha = dict()
    # alpha_start = 0.1
    # alpha_end = 0.4
    # for row in range(dimension):
    #     for col in range(dimension):
    #         alpha[(row, col)] = alpha_start + (col/(dimension-1))*(alpha_end-alpha_start)
    # sim = LatticeForest(dimension, alpha=alpha)
    # sim.rng = seed
    # sim.reset()
    #
    # observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, epsilon)
    # print('median observation accuracy: %0.2f' % (np.median(observation_accuracy)*100))
    # print('median filter accuracy: %0.2f' % (np.median(filter_accuracy)*100))
    # print('{0:0.2f}, {1:0.2f}, {2:0.2f}, {3:0.2f}'.format(np.amin(filter_accuracy), np.median(filter_accuracy),
    #                                                       np.mean(filter_accuracy), np.amax(filter_accuracy)))
    # print('average update time: %0.4fs' % (np.mean(time_data)))

    # the following function will run many simulations and write the results to file
    benchmark(sys.argv)
