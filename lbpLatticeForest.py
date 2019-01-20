import itertools
import numpy as np
import os
import sys

sys.path.append(os.getcwd() + '/lbp')
from factor_graph import FactorGraph
from factors import Factor

from Observe import get_forest_observation, tree_observation_probability


def node2index(dims, row, col, time):
    return time * np.prod(dims) + row*dims[1] + col


class LBP:
    def __init__(self, simulation, prior, iteration_limit=1, horizon=3):
        self.G = FactorGraph(numVar=0, numFactor=0)
        self.factorIdx = 0
        self.time = 0
        self.observation_history = []
        self.prior = prior

        self.add_new_layer(simulation, self.prior)
        self.observation_history += [self.prior]

        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2
        self.tree_state_space = [self.healthy, self.on_fire, self.burnt]

        self.iteration_limit = iteration_limit
        self.horizon = horizon

    def add_new_layer(self, simulation, observation, prior=None):
        if self.time == 0 and prior is None:
            tree_factors = 0
        else:
            tree_factors = np.prod(simulation.dims)
        obs_factors = np.prod(simulation.dims)
        factor_count = tree_factors + obs_factors

        self.G.varName += [[] for _ in range(tree_factors)]
        self.G.domain += [self.tree_state_space for _ in range(tree_factors)]
        self.G.varToFactor += [[] for _ in range(tree_factors)]
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
            self.G.factors.apend(Factor(scope=scope, card=card, val=val, name=name))

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

            obs = self.observation_history[-self.horizon:]

            self.G = FactorGraph(numVar=0, numFactor=0)
            self.factorIdx = 0
            self.time = 0

            self.add_new_layer(obs[0], prior)
            for i in range(1, self.horizon):
                self.add_new_layer(obs[i])

            self.G.runParallelLoopyBP(self.iteration_limit)

        return self.query_belief(simulation, self.time-1)

    def query_belief(self, simulation, time):
        belief = []

        for row in range(simulation.dims[0]):
            row_belief = []
            for col in range(simulation.dims[1]):
                var_idx = node2index(simulation.dims, row, col, time)
                belief = self.G.estimateMarginalProbability(var_idx)
                row_belief += [belief]
            belief += [row_belief]

        return belief

if __name__ == '__main__':
    pass
