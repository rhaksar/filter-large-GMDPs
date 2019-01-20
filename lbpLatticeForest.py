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
from fires.LatticeForest import LatticeForest

from Observe import get_forest_observation, tree_observation_probability


def node2index(dims, row, col, time):
    return time*np.prod(dims) + row*dims[1] + col


class LBP:
    def __init__(self, simulation, observation, prior, iteration_limit=1, horizon=3):
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2

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
        num_nodes = np.prod(simulation.dims)
        if self.time == 0 and prior is None:
            node_factors = 0
        else:
            node_factors = np.prod(simulation.dims)
        obs_factors = np.prod(simulation.dims)
        factor_count = node_factors + obs_factors

        self.G.varName += [[] for _ in range(num_nodes)]
        self.G.domain += [element.state_space for element in simulation.forest.values()]
        self.G.varToFactor += [[] for _ in range(num_nodes)]
        self.G.factorToVar += [[] for _ in range(factor_count)]

        for element in simulation.forest.values():
            row, col = element.position
            var_idx = node2index(simulation.dims, row, col, self.time)
            xname = 'x' + str(row) + str(col) + str(self.time)

            self.G.var.append(var_idx)
            self.G.varName[var_idx] = xname

            scope = [var_idx]
            card = [len(element.state_space)]
            val = np.zeros(card)

            for x in element.state_space:
                val[x] = tree_observation_probability(x, observation[row, col])

            name = 'g_%i%i%i' % (row, col, self.time)
            self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

            self.G.varToFactor[var_idx] += [self.factorIdx]
            self.G.factorToVar[self.factorIdx] += [var_idx]
            self.factorIdx += 1

            if self.time == 0 and prior is not None:
                scope = [var_idx]
                card = [len(element.state_space)]
                val = prior[row, col]

                name = 'b_%i%i%i' % (row, col, self.time)
                self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                self.G.varToFactor[var_idx] += [self.factorIdx]
                self.G.factorToVar[self.factorIdx] += [var_idx]
                self.factorIdx += 1

            elif self.time > 0:
                scope = [node2index(simulation.dims, row, col, self.time-1)]
                for n in element.neighbors:
                    scope += [node2index(simulation.dims, n[0], n[1], self.time-1)]
                scope += [var_idx]

                card = [len(element.state_space) for _ in range(len(scope))]
                val = np.zeros(card)
                iterate = [element.state_space for _ in range(len(scope))]
                for combo in itertools.product(*iterate):
                    x_tm1 = combo[0]
                    f = combo[1:-1].count(self.on_fire)
                    x_t = combo[-1]

                    val[combo] = element.dynamics((x_tm1, f, x_t), (0, 0))

                name = 'f_%i%i%i' % (row, col, self.time-1)
                self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                for idx in scope:
                    self.G.varToFactor[idx] += [self.factorIdx]

                self.G.factorToVar[self.factorIdx] += scope
                self.factorIdx += 1

        self.time += 1

    def filter(self, simulation, observation):
        tic = time.clock()
        self.observation_history += [observation]

        if 1 <= self.time < self.horizon:
            self.add_new_layer(simulation, observation)
            status = self.G.runParallelLoopyBP(self.iteration_limit)

        else:
            prior = self.query_belief(simulation, 1)

            obs_hist = self.observation_history[-self.horizon:]

            self.G = FactorGraph(numVar=0, numFactor=0)
            self.factorIdx = 0
            self.time = 0

            self.add_new_layer(simulation, obs_hist[0], prior)
            for i in range(1, self.horizon):
                self.add_new_layer(simulation, obs_hist[i])

            status = self.G.runParallelLoopyBP(self.iteration_limit)

        toc = time.clock()
        return self.query_belief(simulation, self.time-1), status, toc-tic

    def query_belief(self, simulation, time):
        belief = np.zeros((simulation.dims[0], simulation.dims[1], 3))

        for row in range(simulation.dims[0]):
            for col in range(simulation.dims[1]):
                var_idx = node2index(simulation.dims, row, col, time)
                b = self.G.estimateMarginalProbability(var_idx)
                belief[row, col] = b

        return belief


def run_simulation(sim_obj, iteration_limit, horizon):
    num_trees = np.prod(sim_obj.dims)
    control = defaultdict(lambda: (0, 0))

    # initial belief
    belief = np.zeros((sim_obj.dims[0], sim_obj.dims[1], 3))
    state = sim_obj.dense_state()
    p = np.where(state == 0)
    belief[p[0], p[1], :] = [1, 0, 0]
    p = np.where(state == 1)
    belief[p[0], p[1], :] = [0, 1, 0]
    p = np.where(state == 2)
    belief[p[0], p[1], :] = [0, 0, 1]
    belief = belief / belief.sum(axis=2, keepdims=True)

    robot = LBP(sim_obj, sim_obj.dense_state(), belief, iteration_limit=iteration_limit, horizon=horizon)

    observation_acc = []
    filter_acc = []
    update_time = []

    while not sim_obj.end:
        sim_obj.update(control)
        state = sim_obj.dense_state()

        obs = get_forest_observation(sim_obj)
        obs_acc = np.sum(obs == state)/num_trees

        belief, status, timing = robot.filter(sim_obj, obs)
        estimate = np.argmax(belief, axis=2)
        f_acc = np.sum(estimate == state)/num_trees
        update_time.append(timing)

        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    return observation_acc, filter_acc, update_time


if __name__ == '__main__':
    dimension = 3
    Kmax = 1
    H = 3
    total_sims = 20

    results = dict()
    results['dimension'] = dimension
    results['Kmax'] = Kmax
    results['horizon'] = H
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

        observation_accuracy, filter_accuracy, time_data = run_simulation(sim, Kmax, H)
        results[seed] = dict()
        results[seed]['observation_accuracy'] = observation_accuracy
        results[seed]['LBP_accuracy'] = filter_accuracy
        results[seed]['time_per_update'] = time_data

        if (s+1) % 10 == 0 and (s+1) != total_sims:
            st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            print('[%s] finished %d simulations' % (st, s+1))

            filename = '[SAVE] ' + 'lbp_d' + str(dimension) + \
                       '_Kmax' + str(Kmax) + '_h' + str(H) + \
                       '_s' + str(s+1) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()

    st = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] finish' % st)
    t1 = time.clock()
    print('%0.2fs = %0.2fm = %0.2fh elapsed' % (t1-t0, (t1-t0)/60, (t1-t0)/(60*60)))

    filename = 'lbp_d' + str(dimension) + \
               '_Kmax' + str(Kmax) + '_h' + str(H) + \
               '_s' + str(total_sims) + '.pkl'
    output = open(filename, 'wb')
    pickle.dump(results, output)
    output.close()

