from collections import defaultdict
import itertools
import numpy as np
import os
import sys

sys.path.append(os.getcwd() + '/lbp')
sys.path.append(os.getcwd() + '/simulators')
from factor_graph import FactorGraph
from factors import Factor
from fires.LatticeForest import LatticeForest

from Observe import get_forest_observation, tree_observation_probability


def node2index(dims, row, col, time):
    return time * np.prod(dims) + row*dims[1] + col


class LBP:
    def __init__(self, simulation, observation, prior, iteration_limit=1, horizon=3):
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2
        self.tree_state_space = [self.healthy, self.on_fire, self.burnt]

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
        tree_nodes = np.prod(simulation.dims)
        if self.time == 0 and prior is None:
            tree_factors = 0
        else:
            tree_factors = np.prod(simulation.dims)
        obs_factors = np.prod(simulation.dims)
        factor_count = tree_factors + obs_factors

        self.G.varName += [[] for _ in range(tree_nodes)]
        self.G.domain += [self.tree_state_space for _ in range(tree_nodes)]
        self.G.varToFactor += [[] for _ in range(tree_nodes)]
        self.G.factorToVar += [[] for _ in range(factor_count)]

        for element in simulation.forest.values():
            row, col = element.position
            var_idx = node2index(simulation.dims, row, col, self.time)
            xname = 'x' + str(row) + str(col) + str(self.time)

            self.G.var.append(var_idx)
            self.G.varName[var_idx] = xname

            scope = [var_idx]
            card = [len(self.tree_state_space)]
            val = np.zeros(card)

            for x in self.tree_state_space:
                val[x] = tree_observation_probability(x, observation[row, col])

            name = 'g_%i%i%i' % (row, col, self.time)
            self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

            self.G.varToFactor[var_idx] += [self.factorIdx]
            self.G.factorToVar[self.factorIdx] += [var_idx]
            self.factorIdx += 1

            if self.time == 0 and prior is not None:
                scope = [var_idx]
                card = [len(self.tree_state_space)]
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

                card = [len(self.tree_state_space) for _ in range(len(scope))]
                val = np.zeros(card)
                iterate = [self.tree_state_space for _ in range(len(scope))]
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

        self.observation_history += [observation]

        if 1 <= self.time < self.horizon:
            self.add_new_layer(simulation, observation)
            self.G.runParallelLoopyBP(self.iteration_limit)

        else:
            prior = self.query_belief(simulation, 1)

            obs_hist = self.observation_history[-self.horizon:]

            self.G = FactorGraph(numVar=0, numFactor=0)
            self.factorIdx = 0
            self.time = 0

            self.add_new_layer(simulation, obs_hist[0], prior)
            for i in range(1, self.horizon):
                self.add_new_layer(simulation, obs_hist[i])

            self.G.runParallelLoopyBP(self.iteration_limit)

        return self.query_belief(simulation, self.time-1)

    def query_belief(self, simulation, time):
        belief = np.zeros((simulation.dims[0], simulation.dims[1], len(self.tree_state_space)))

        for row in range(simulation.dims[0]):
            for col in range(simulation.dims[1]):
                var_idx = node2index(simulation.dims, row, col, time)
                b = self.G.estimateMarginalProbability(var_idx)
                belief[row, col] = b

        return belief


if __name__ == '__main__':
    dimension = 3
    control = defaultdict(lambda: (0, 0))
    seed = 1000
    np.random.seed(seed)

    sim = LatticeForest(dimension, rng=seed)
    num_trees = np.prod(sim.dims)

    # initial belief
    belief = np.zeros((sim.dims[0], sim.dims[1], 3))
    state = sim.dense_state()
    p = np.where(state == 0)
    belief[p[0], p[1], :] = [1, 0, 0]
    p = np.where(state == 1)
    belief[p[0], p[1], :] = [0, 1, 0]
    p = np.where(state == 2)
    belief[p[0], p[1], :] = [0, 0, 1]
    belief = belief / belief.sum(axis=2, keepdims=True)

    robot = LBP(sim, sim.dense_state(), belief, iteration_limit=1, horizon=3)

    observation_acc = []
    filter_acc = []
    for _ in range(5):
        sim.update(control)
        state = sim.dense_state()

        obs = get_forest_observation(sim)
        obs_acc = np.sum(obs == state)

        belief = robot.filter(sim, obs)
        estimate = np.argmax(belief, axis=2)
        f_acc = np.sum(estimate == state)
        print('observation/filter accuracy: %0.2f / %0.2f' % (obs_acc*100/num_trees, f_acc*100/num_trees))

        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    print('observation, filter min accuracy: %0.2f, %0.2f' % (np.amin(observation_acc)*100/num_trees,
                                                              np.amin(filter_acc)*100/num_trees))
    print('observation, filter median accuracy: %0.2f, %0.2f' % (np.median(observation_acc)*100/num_trees,
                                                                 np.median(filter_acc)*100/num_trees))
    print('observation, filter max accuracy: %0.2f, %0.2f' % (np.amax(observation_acc)*100/num_trees,
                                                              np.amax(filter_acc)*100/num_trees))
    print()

