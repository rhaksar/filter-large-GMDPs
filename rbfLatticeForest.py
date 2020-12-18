from collections import defaultdict
import copy
import numpy as np
import operator

from simulators.fires.LatticeForest import LatticeForest
from filters.observe import get_forest_observation, tree_observation_probability


def reachable_states(input_state, tree_object, forest_dimension):
    """
    Efficiently calculate the possible next states with non-zero probability using recursion.

    :param input_state: list of length forest_dimension*forest_dimension where each element indicates the state of the
    corresponding tree in the forest. A 'flat' representation is used by flattening the matrix representing the state of
    trees in the forest using row-major order.
    :param tree_object: a Tree object from the LatticeForest simulator, to calculate the transition probability for
    different scenarios.
    :param forest_dimension: int representing the size of one side of the square LatticeForest.
    :return: a dictionary where the keys are possible next states (as strings) and their corresponding probability.
    """
    num_trees = len(input_state)
    assert num_trees == forest_dimension**2

    healthy, on_fire, burnt = 0, 1, 2

    def generate(reachable, next_state=input_state, element=0, probability=1):

        if element < num_trees:

            if input_state[element] == healthy:
                # if a given element is healthy, find out how many of its neighbors are on fire
                num_neighbors_on_fire = 0

                # unflatten index to (row, col)
                row, col = np.unravel_index(element, (forest_dimension, forest_dimension), order='C')

                # iterate through cardinal directions and re-flatten to check the state value
                for (dr, dc) in [(-1, 0), (+1, 0), (0, +1), (0, -1)]:
                    if 0 <= row+dr < forest_dimension and 0 <= col+dc < forest_dimension:
                        neighbor_index = np.ravel_multi_index([[row+dr], [col+dc]],
                                                              (forest_dimension, forest_dimension),
                                                              order='C')
                        if input_state[np.squeeze(neighbor_index)] == on_fire:
                            num_neighbors_on_fire += 1

                # if there are no neighbors on fire, the tree must remain healthy, so move on to the next element
                if num_neighbors_on_fire == 0:
                    generate(reachable, next_state=next_state, element=element+1, probability=probability)
                    return

                # otherwise, the tree may transition to on fire ...
                modified_state = copy.copy(next_state)
                modified_state[element] = on_fire
                modified_probability = probability*tree_object.dynamics((healthy, num_neighbors_on_fire, on_fire))
                generate(reachable, modified_state, element=element+1, probability=modified_probability)

                # or the tree may remain healthy
                modified_probability = probability*tree_object.dynamics((healthy, num_neighbors_on_fire, healthy))
                generate(reachable, next_state=next_state, element=element+1, probability=modified_probability)
                return

            elif input_state[element] == on_fire:
                # a tree on fire either burns out ...
                modified_state = copy.copy(next_state)
                modified_state[element] = burnt
                modified_probability = probability*tree_object.dynamics((on_fire, -1, burnt))
                generate(reachable, next_state=modified_state, element=element+1, probability=modified_probability)

                # or the tree may remain on fire
                modified_probability = probability*tree_object.dynamics((on_fire, -1, on_fire))
                generate(reachable, next_state=next_state, element=element+1, probability=modified_probability)

            elif input_state[element] == burnt:
                # a burnt tree must remain burnt
                generate(reachable, next_state=next_state, element=element+1, probability=probability)
                return

            else:
                raise Exception('invalid state value encountered')

        else:
            # after iterating through all elements in the state, store the next state and its probability
            state_string = ''.join(str(e) for e in next_state)
            reachable[state_string] = probability
            return

    result = dict()
    generate(result)
    return result


def rbf(belief, observation, tree_object, forest_dimension):
    """
    Function to implement the one-step update for the recursive Bayesian filter.

    :param belief: dictionary representing the current belief, each key is a flat state (string) and the corresponding
    value is the state probability.
    :param observation: a numpy 2D array representing the observation for each tree.
    :param tree_object: a Tree object to simplify implementing the transition probabilities.
    :param forest_dimension: int representing the size of one side of the square LatticeForest.
    :return: dictionary representing the updated belief.
    """
    num_trees = forest_dimension**2

    next_belief = dict()

    # first, apply the dynamics to produce an open-loop belief (no measurement correction)
    dynamics_update = defaultdict(lambda: 0)
    for state_idx in range(3**num_trees):  # iterating through all states will be slow
        current_state = np.base_repr(state_idx, base=3).zfill(num_trees)

        # only consider reachable (non-zero probability) states, to improve speed
        current_state_list = np.array(list(current_state), dtype=int)
        reachable = reachable_states(current_state_list, tree_object, forest_dimension)

        for next_state in reachable:
            dynamics_update[next_state] += reachable[next_state]*belief[current_state]

    # second, apply measurement to correct belief
    normalization = 0
    for state_idx in range(3**num_trees):
        next_state = np.base_repr(state_idx, base=3).zfill(num_trees)

        observation_prob = 1
        for tree_idx in range(num_trees):
            row, col = np.unravel_index(tree_idx, (forest_dimension, forest_dimension), order='C')
            observation_prob *= tree_observation_probability(int(next_state[tree_idx]), observation[row, col])

        next_belief[next_state] = observation_prob*dynamics_update[next_state]
        normalization += next_belief[next_state]

    # normalize belief
    for next_state in next_belief:
        next_belief[next_state] /= normalization

    return next_belief


def run_simulation(sim_object, forest_dimension):
    """
    Run a single simulation with RBF and report the filter accuracy.

    :param sim_object: LatticeForest simulation object.
    :param forest_dimension: int representing the size of one side of the square LatticeForest.
    :return: tuple of (observation accuracy, filter_accuracy), each of which is a list containing the time-history of
    accuracies for the observation and using the RBF, respectively.
    """
    num_trees = np.squeeze(np.prod(sim_object.dims))
    assert num_trees == forest_dimension**2
    tree = sim_object.group[(0, 0)]

    # exact initial belief
    belief = dict()
    dense_state = sim_object.dense_state()
    for state_idx in range(3**num_trees):
        state = np.base_repr(state_idx, base=3).zfill(num_trees)
        belief[state] = 0.0
    state = ''.join([str(e) for e in dense_state.reshape(num_trees, order='C')])
    belief[state] = 1.0

    observation_acc = []
    filter_acc = []

    while not sim_object.end:

        sim_object.update()
        state = sim_object.dense_state()

        obs = get_forest_observation(sim_object)
        obs_acc = np.sum(obs == state)/num_trees

        belief = rbf(belief, obs, tree, forest_dimension)

        # compare maximum likelihood state from RBF to ground truth
        max_likelihood_state = max(belief.items(), key=operator.itemgetter(1))[0]
        max_likelihood_state = np.array(list(max_likelihood_state), dtype=int).reshape((forest_dimension,
                                                                                        forest_dimension),
                                                                                       order='C')
        f_acc = np.sum(max_likelihood_state == state)/num_trees

        observation_acc.append(obs_acc)
        filter_acc.append(f_acc)

    return observation_acc, filter_acc


if __name__ == '__main__':
    dimension = 3
    sim = LatticeForest(dimension)

    # running the RBF is slow, even for small LatticeForest sizes
    observation, filter = run_simulation(sim, dimension)
    print('median observation accuracy = {0:0.2f}'.format(np.median(observation)))
    print('median filter accuracy = {0:0.2f}'.format(np.median(filter)))
