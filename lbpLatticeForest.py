from collections import defaultdict
from datetime import datetime
import itertools
import numpy as np
import pickle
import time

from simulators.fires.LatticeForest import LatticeForest
from filters.lbp import LBP
from filters.observe import get_forest_observation, tree_observation_probability


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
