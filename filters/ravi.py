import copy
import numpy as np
import time


# helper function to carefully multiply probabilities
def multiply_probabilities(values, threshold):
    if any([v < threshold for v in values]):
        return 0
    else:
        sum_log = sum([np.log(v) for v in values])
        if sum_log <= np.log(threshold):
            return 0
        else:
            return np.exp(sum_log)


# linear approximation to the logarithm function over the interval [epsilon, 1]
# inputs less than epsilon are rounded up to epsilon
def approximation(x, constant, epsilon):
    return np.asarray([constant*(1-p) if p > epsilon else constant*(1-epsilon) for p in x])


class RAVI(object):
    """
    Relaxed Anonymous Variational Inference, tailored to the simulators package; more details at
    https://github.com/rhaksar/simulators.
    """
    def __init__(self, belief, iteration_limit=1, epsilon=1e-10):
        """
        Initialize filter object.

        :param belief: dictionary representing the initial belief, each key refers to an node/vertex in the graph and
        the value is a numpy array describing the probabilities of different states.
        :param iteration_limit: message-passing iteration limit.
        :param epsilon: threshold for smallest non-zero probability.
        """
        self.belief = belief

        self.iteration_limit = iteration_limit
        self.epsilon = epsilon
        self.constant = np.log(epsilon)/(1-epsilon)

    def filter(self, simulation, build_candidate):
        """
        Produce posterior factors to update belief.

        :param simulation: simulation object for a graph-based process.
        :param build_candidate: function handle which creates an intermediate distribution for each node/vertex in the
        graph, and is specific to a particular graph-based process.

        :returns posterior: dictionary representing the posterior belief, each key is a node/vertex and refers to a
        numpy array describing the probabilities of different states.
        :returns status: tuple of (status (string), iterations (int)). status is either 'Converged' or 'Cutoff'
        indicating if the filter converged given the iteration limit. if the filter converges, iterations indicates the
        number of required message-passing iterations.
        :returns time: time taken to update the belief.
        """
        tic = time.clock()
        filter_graph = dict()  # structure to hold data for message-passing algorithm

        # initialize posterior as prior belief
        posterior = copy.copy(self.belief)

        # initialize message for each element as the prior belief
        for key in simulation.group.keys():
            filter_graph[key] = dict()
            filter_graph[key]['message'] = self.belief[key]

        # perform message-passing to update posterior factors
        status = ['Cutoff', self.iteration_limit]
        for iteration in range(self.iteration_limit):
            num_changed = 0

            for key in simulation.group.keys():
                element = simulation.group[key]

                # build candidate message based on transition model, observation model, and neighbors' messages
                filter_graph[key]['candidate'] = build_candidate(element, filter_graph, self.epsilon)

                # update posterior factor
                q = np.zeros(len(element.state_space))
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        values = [self.belief[key][s_tm1], filter_graph[key]['candidate'][s_tm1, s_t]]
                        q[s_t] += multiply_probabilities(values, self.epsilon)

                q_approx = approximation(q, self.constant, self.epsilon)

                # normalize posterior factor
                normalization = 0
                for idx, v in enumerate(q_approx):
                    if v <= np.log(self.epsilon):
                        q_approx[idx] = 0
                        continue

                    q_approx[idx] = np.exp(v)
                    normalization += q_approx[idx]
                q_approx /= normalization
                q = q_approx

                # check if the max likelihood estimate changed
                if np.argmax(q) != np.argmax(posterior[key]):
                    num_changed += 1
                posterior[key] = q

                # calculate next message
                filter_graph[key]['next_message'] = np.zeros(len(element.state_space))
                for s_tm1 in element.state_space:
                    for s_t in element.state_space:
                        values = [q[s_t], filter_graph[key]['candidate'][s_tm1, s_t]]
                        filter_graph[key]['next_message'][s_tm1] += multiply_probabilities(values, self.epsilon)

                    values = [self.belief[key][s_tm1], filter_graph[key]['next_message'][s_tm1]]
                    filter_graph[key]['next_message'][s_tm1] = multiply_probabilities(values, self.epsilon)

                # normalize next message
                d = filter_graph[key]['next_message']
                normalization = np.sum(d)
                d /= normalization
                filter_graph[key]['next_message'] = d

            # if less than 1% of nodes are changing their max likelihood estimate, break
            if iteration > 1 and num_changed/np.prod(simulation.dims) <= 0.01:
                status = ['Converged', iteration]
                break

            # update messages for next round
            for key in simulation.group.keys():
                filter_graph[key]['message'] = filter_graph[key]['next_message']

        toc = time.clock()
        self.belief = posterior
        return posterior, status, toc-tic
