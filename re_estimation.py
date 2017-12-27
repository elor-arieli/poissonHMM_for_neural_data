import numpy as np


def update_pi_array(gammas):
    return gammas[0,:]


def update_Aij_matrix(zettas, gammas):
    num_of_states = gammas.shape[1]
    Aij_matrix = np.zeros((num_of_states,num_of_states))

    for state_i in range(num_of_states):
        for state_j in range(num_of_states):
            Aij_matrix[state_i,state_j] = zettas[:-1,state_i,state_j].sum() / gammas[:-1,state_i].sum()

    return Aij_matrix


def update_B_matrix(alphas, bettas, neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    # alphas and bettas are matrices where axis 0 - time and 1 - states
    time_points = neural_data_matrix.shape[2]
    num_of_states = alphas.shape[1]
    num_of_neurons = neural_data_matrix.shape[1]
    B_matrix = np.zeros((num_of_states, num_of_neurons)) # axis 0 is the state and axis 1 is the neuron

    for neuron in range(num_of_neurons):
        for state in range(num_of_states):
            B_matrix[state,neuron] = np.inner(np.multiply(alphas[:-1,state],bettas[:-1,state]),
                                              neural_data_matrix.mean(axis=0)[neuron,1:]) \
                                     / np.inner(alphas[:,state],bettas[:,state])

    return B_matrix
