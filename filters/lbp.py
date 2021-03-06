import itertools
import numpy as np
import time

from filters.factor_graph import FactorGraph
from filters.factors import Factor


# function to map a node in the graphical model to a numerical index
def node2index(dims, idx, time):
    return time*np.prod(dims) + idx


class LBP:
    """
    Implementation of loopy belief propagation, tailored to the simulators package; see details at
    https://github.com/rhaksar/simulators.
    """
    def __init__(self, simulation, observation, observation_model, transition_model, prior,
                 iteration_limit=1, horizon=3):
        """
        Initialize filter object.

        :param simulation: simulator object for a graph-based process.
        :param observation: observation at initial time step.
        :param observation_model: function handle describing the observation model for a single node/vertex in the
        graph-based process. used to build the initial layer of the graphical model in the filter. arguments are
        (node/vertex object, state value, observation value).
        :param transition_model: function handle describing the transition model for a single node/vertex in the
        graph-based process. used to build the initial layer of the graphical model in the filter.  arguments are
        (node/vertex object, state values of object and neighbor objects).
        :param prior: initial belief of the process.
        :param iteration_limit: maximum number of message-passing iterations.
        :param horizon: maximum number of layers in the graphical model. when adding a layer would exceed this limit,
        prior layers are dropped.
        """
        self.G = FactorGraph(numVar=0, numFactor=0)
        self.factorIdx = 0
        self.time = 0

        self.observation_history = []
        self.prior = prior

        self.add_new_layer(simulation, observation, observation_model, transition_model, self.prior)
        self.observation_history += [observation]

        self.iteration_limit = iteration_limit
        self.horizon = horizon

    def add_new_layer(self, simulation, observation, observation_model, transition_model, prior=None):
        """
        Adds a new time layer to the graph corresponding to the new measurement y. If prior is given and
        self.time = 0 then the layer also includes factors corresponding to the priors.

        First, this function builds the nodes for each new simulation element at the current time.
        Then, it adds factors for measurements. If self.time > 0 then factors are added for the dynamics.

        :param simulation: simulator object for a graph-based process.
        :param observation: observation for the current time step.
        :param observation_model: function handle describing the observation model for a single node/vertex in the
        graph-based process. arguments are (node/vertex object, state value, observation value).
        :param transition_model: function handle describing the transition model for a single node/vertex in the
        graph-based process. arguments are (node/vertex object, state values of object and neighbor objects).
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

        # add variables and factors for each element
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
                val[x] = observation_model(element, x, observation[key])

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

                # iterate through all combination of states for the factor and the other connected factors
                for combo in itertools.product(*iterate):
                    val[combo] = transition_model(element, combo)

                name = 'f_%i%i' % (element.numeric_id, self.time-1)
                self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                for idx in scope:
                    # connect this new factor to each variable's list
                    self.G.varToFactor[idx] += [self.factorIdx]

                # connect this variable to factor's list
                self.G.factorToVar[self.factorIdx] += scope
                self.factorIdx += 1

        self.time += 1

    def filter(self, simulation, observation, observation_model, transition_model):
        """
        Method to build the graphical model and generate the belief.

        :param simulation: simulator object for a graph-based process.
        :param observation: observation for the current time step.
        :param observation_model: function handle describing the observation model for a single node/vertex in the
        graph-based process. arguments are (node/vertex object, state value, observation value).
        :param transition_model: function handle describing the transition model for a single node/vertex in the
        graph-based process. arguments are (node/vertex object, state values of object and neighbor objects).

        :return belief: updated belief for last layer in the filter's graphical model.
        :return status: tuple of (status, iterations), indicating whether or not the filter converged ('Converged' or
        'Cutoff') and the number of iterations taken (which will be at most the message-passing iteration limit).
        :return time: time taken for the filter to update the belief.
        """
        tic = time.clock()
        self.observation_history += [observation]

        # case where graphical model has not grown too large
        if 1 <= self.time < self.horizon:
            # add layer for measurement
            self.add_new_layer(simulation, observation, observation_model, transition_model)
            status = self.G.runParallelLoopyBP(self.iteration_limit)

        # case where graphical model is too large so rebuild
        else:
            # get prior at time 1
            prior = self.query_belief(simulation, 1)

            obs_hist = self.observation_history[-self.horizon:]

            # create new graph
            self.G = FactorGraph(numVar=0, numFactor=0)
            self.factorIdx = 0
            self.time = 0

            # create first layer with measurement and priors
            self.add_new_layer(simulation, obs_hist[0], observation_model, transition_model, prior)

            # create remaining layers with measurements
            for i in range(1, self.horizon):
                self.add_new_layer(simulation, obs_hist[i], observation_model, transition_model)

            status = self.G.runParallelLoopyBP(self.iteration_limit)

        toc = time.clock()
        return self.query_belief(simulation, self.time-1), status, toc-tic

    def query_belief(self, simulation, time):
        """
        Method to get belief from graphical model. Returns a dictionary where each key is the name of a simulation
        element and values correspond to a list with a probability for each element state.

        :param simulation: simulation object for a graph-based process.
        :param time: time (i.e., layer) at which to query the belief.

        :return belief: maximum-likelihood belief as a dictionary, each key is an ID referring to a node/vertex
        (defined by the simulation object) and refers to the maximum-likelihood state value.
        """
        belief = {}

        for key in simulation.group.keys():
            var_idx = node2index(simulation.dims, simulation.group[key].numeric_id, time)
            b = self.G.estimateMarginalProbability(var_idx)
            belief[key] = b

        return belief
