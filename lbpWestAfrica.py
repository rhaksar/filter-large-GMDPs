from datetime import datetime
import numpy as np
import pickle
import time

from simulators.epidemics.WestAfrica import WestAfrica
from filters.lbp import LBP
from filters.observe import get_ebola_observation, region_observation_probability


# functions to connect LBP interface to observation and transition models for Regions
def obs_model(region, state_value, obs_value):
    return region_observation_probability(state_value, obs_value)


def trans_model(region, combo):
    x_tm1 = combo[0]
    f = combo[1:-1].count(region.infected)
    x_t = combo[-1]
    return region.dynamics((x_tm1, f, x_t))


def run_simulation(sim_obj, iteration_limit, horizon):
    """
    Function to run a single simulation of the WestAfrica simulator using loopy belief propagation.
    """
    num_regions = np.prod(sim_obj.dims)

    # initial belief - exact state
    belief = {}
    state = sim_obj.dense_state()
    for key in state.keys():
        belief[key] = [0, 0, 0]
        belief[key][state[key]] = 1

    # instantiate filter
    robot = LBP(sim_obj, sim_obj.dense_state(), obs_model, trans_model, belief,
                iteration_limit=iteration_limit, horizon=horizon)

    observation_acc = []
    filter_acc = []
    update_time = []

    # run for only 75 iterations as process is non self-terminating
    for _ in range(75):
        # update simulator and get state
        sim_obj.update()
        state = sim_obj.dense_state()

        # get observation and observation accuracy
        obs = get_ebola_observation(sim_obj)
        obs_acc = [obs[name] == state[name] for name in state.keys()].count(True)/num_regions

        # run filter and get belief
        belief, status, timing = robot.filter(sim_obj, obs, obs_model, trans_model)
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

    if len(arguments) > 1:
        Kmax = int(arguments[1][1:])

    print('[LBP] Kmax = {0:d}'.format(Kmax))
    print('running for {0:d} simulation(s)'.format(total_sims))

    # dictionary for storing results for each simulation
    results = dict()
    results['Kmax'] = Kmax
    results['horizon'] = H
    results['total_sims'] = total_sims

    # set initial condition
    outbreak = {('guinea', 'gueckedou'): 1}
    sim = WestAfrica(outbreak)

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[{0}] start'.format(st))

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
            print('[{0}] finished {1:d} simulations'.format(st, s+1))

            filename = '[SAVE] ' + 'lbp_wa' + \
                       '_Kmax' + str(Kmax) + '_h' + str(H) + \
                       '_s' + str(s + 1) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()

    # save full results to file
    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[{0}] finish'.format(st))
    t1 = time.clock()
    print('{0:0.2f}s = {1:0.2f}m = {2:0.2f}h elapsed'.format(t1-t0, (t1-t0)/60, (t1-t0)/(60*60)))

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

    print('[LBP] Kmax = {0:d}'.format(Kmax))

    # set initial condition
    outbreak = {('guinea', 'gueckedou'): 1}
    sim = WestAfrica(outbreak)

    observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, H)
    print('median observation accuracy: {0:0.2f}'.format(np.median(observation_accuracy)*100))
    print('median filter accuracy: {0:0.2f}'.format(np.median(filter_accuracy)*100))
    print('average update time: {0:0.4f}s'.format(np.mean(time_data)))

    # the following function will run the filter for many simulations and write the results to file
    # benchmark(sys.argv)
