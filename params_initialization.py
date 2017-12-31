import numpy as np
import random


def create_transition_matrix(num_of_states, type='shenoy'):
    transition_matrix = np.zeros((num_of_states,num_of_states))
    diag = random.uniform(0.85, 0.9)
    if type == "FX":
        for i in range(num_of_states):
            for j in range(num_of_states):
                if i == j:
                    transition_matrix[i,j] = diag
                elif i == j-1:
                    transition_matrix[i, j] = 1-diag
        return transition_matrix

    elif type == "F":
        for i in range(num_of_states):
            for j in range(num_of_states):
                if i == j:
                    transition_matrix[i,j] = diag
                elif i < j:
                    transition_matrix[i, j] = (1 - diag) / sum(np.arange(num_of_states) - i)*(num_of_states - j + 1)
        return transition_matrix

    elif type == "ATA":
        for i in range(num_of_states):
            for j in range(num_of_states):
                if i == j:
                    transition_matrix[i, j] = diag
                elif i < j:
                    transition_matrix[i, j] = (1 - diag) / (num_of_states - 1)
        return transition_matrix

    elif type == "shenoy":
        nd = 1-diag
        nd_8 = nd/8.0
        nd_2 = nd/2.0
        transition_matrix = [[diag, nd_8, nd_8, nd_8, nd_8, nd_8, nd_8, nd_8, nd_8, 0,    0],
                             [nd_8, diag, nd_8, nd_8, nd_8, nd_8, nd_8, nd_8, nd_8, 0,    0],
                             [nd_8, nd_8, diag, nd_8, nd_8, nd_8, nd_8, nd_8, nd_8, 0,    0],
                             [nd_8, nd_8, nd_8, diag, nd_8, nd_8, nd_8, nd_8, nd_8, 0,    0],
                             [nd_8, nd_8, nd_8, nd_8, diag, nd_8, nd_8, nd_8, nd_8, 0,    0],
                             [0,    0,    0,    0,    0,    diag, 0,    0,    0,    nd_2, nd_2],
                             [0,    0,    0,    0,    0,    0,    diag, 0,    0,    nd_2, nd_2],
                             [0,    0,    0,    0,    0,    0,    0,    diag, 0,    nd_2, nd_2],
                             [0,    0,    0,    0,    0,    0,    0,    0,    diag, nd_2, nd_2],
                             [0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0],
                             [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1]]
        return np.array(transition_matrix)

    else:
        raise SyntaxError("type is not valid, should be shenoy, F, FX or ATA")

def create_B_matrix_poissonian_rates(neural_data_matrix):
    # return a matrix of states x neurons with the firing rate in every state.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    BL_state = neural_data_matrix[:,:,:20].mean(axis=2).mean(axis=0)
    BL_state_1 = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))
    BL_state_2 = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))
    BL_state_3 = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))
    BL_state_4 = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))
    BL_state_5 = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))

    water_state = neural_data_matrix[:18,   :, 20:40].mean(axis=2).mean(axis=0)
    water_state = water_state + np.random.normal(loc=water_state.mean(), scale=1.0, size=len(water_state))

    # sugar_state = neural_data_matrix[18:36, :, 20:40].mean(axis=2).mean(axis=0)
    # sugar_state = sugar_state + np.random.normal(loc=sugar_state.mean(), scale=1.0, size=len(sugar_state))
    sugar_state = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))

    # nacl_state  = neural_data_matrix[36:54, :, 20:40].mean(axis=2).mean(axis=0)
    # nacl_state = nacl_state + np.random.normal(loc=nacl_state.mean(), scale=1.0, size=len(nacl_state))
    nacl_state = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))

    # ca_state    = neural_data_matrix[54:,   :, 20:40].mean(axis=2).mean(axis=0)
    # ca_state = ca_state + np.random.normal(loc=ca_state.mean(), scale=1.0, size=len(ca_state))
    ca_state = BL_state + np.random.normal(loc=BL_state.mean(), scale=5.0, size=len(BL_state))

    good_palatability = neural_data_matrix[36:54, :, 40:70].mean(axis=2).mean(axis=0)
    good_palatability = good_palatability + np.random.normal(loc=good_palatability.mean(), scale=1.0, size=len(good_palatability))
    bad_palatability = neural_data_matrix[54:,   :, 40:70].mean(axis=2).mean(axis=0)
    bad_palatability = bad_palatability + np.random.normal(loc=bad_palatability.mean(), scale=10.0, size=len(bad_palatability))

    # return a matrix where axis 0 is the state and axis 1 is the neuron
    return np.abs(np.vstack((BL_state_1,BL_state_2,BL_state_3,BL_state_4,BL_state_5,water_state,nacl_state,sugar_state,ca_state,good_palatability,bad_palatability)))

def create_pi_array(num_of_states,start_from="first five BL"):
    pi_array = np.zeros(num_of_states)
    if start_from=="first":
        pi_array[0] = 1
    elif start_from == "first five BL":
        pi_array[0:5] = 0.2
    else:
        pi_array += (1.0/num_of_states)
    return pi_array