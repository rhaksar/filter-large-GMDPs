import copy
import numpy as np
import time


# helper function to carefully multiply probabilities
def multiply_probabilities(values):
    threshold = 1e-20  # determines the smallest non-zero probability
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
def approximation(x, epsilon=1e-10):
    constant = (np.log(epsilon)/(1-epsilon))
    if x < epsilon:
        return constant*(1 - epsilon)
    else:
        return constant*(1 - x)


class RAVI(object):
    """
    Relaxed Anonymous Variational Inference.
    """
    def __init__(self, belief, iteration_limit=1, epsilon=1e-10):
        """
        Initialize filter object.

        :param belief:
        :param iteration_limit:
        :param epsilon:
        """
        self.belief = belief

        self.iteration_limit = iteration_limit
        self.epsilon = epsilon

    def filter(self, simulation, build_candidate):
        """
        Produce posterior factors to update belief.

        :param simulation:
        :param build_candidate:

        :returns posterior:
        :returns status:
        :returns time:
        """
        tic = time.clock()
        filter_graph = dict()  # structure to hold data for message-passing algorithm

        # initialize posterior as current belief
        posterior = copy.copy(self.belief)

        # initialize message for each element
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
                filter_graph[key]['candidate'] = build_candidate(element, filter_graph)

                # update posterior factor
                q = np.zeros(len(element.state_space))
                for s_t in element.state_space:
                    for s_tm1 in element.state_space:
                        values = [self.belief[key][s_tm1], filter_graph[key]['candidate'][s_tm1, s_t]]
                        q[s_t] += multiply_probabilities(values)

                    q[s_t] = approximation(q[s_t], self.epsilon)

                # regularization
                # coeff = 0.6
                # q_mean = np.mean(q)
                # q = [v - coeff*np.sign(v-q_mean)*np.abs(v-q_mean) for v in q]

                # normalize posterior factor
                q = [v-max(q) for v in q]
                normalization = 0
                for idx, v in enumerate(q):
                    if v <= -100:
                        q[idx] = 0
                        continue

                    q[idx] = np.exp(v)
                    normalization += q[idx]

                q /= normalization

                # regularization
                # q += 0.1*np.random.rand(3)
                # q /= np.sum(q)

                # check if the max likelihood estimate changed
                if np.argmax(q) != np.argmax(posterior[key]):
                    num_changed += 1
                posterior[key] = q

                # calculate next message
                filter_graph[key]['next_message'] = np.zeros(len(element.state_space))
                for s_tm1 in element.state_space:
                    for s_t in element.state_space:
                        values = [q[s_t], filter_graph[key]['candidate'][s_tm1, s_t]]
                        filter_graph[key]['next_message'][s_tm1] += multiply_probabilities(values)

                    values = [self.belief[key][s_tm1], filter_graph[key]['next_message'][s_tm1]]
                    filter_graph[key]['next_message'][s_tm1] = multiply_probabilities(values)

                # normalize next message
                d = filter_graph[key]['next_message']
                # d = [np.log(v) for v in d]
                # d = [v-max(d) for v in d]
                # normalization = 0
                # for idx, v in enumerate(d):
                #     if v <= -100:
                #         d[idx] = 0
                #         continue
                #
                #     d[idx] = np.exp(v)
                #     normalization += d[idx]
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
