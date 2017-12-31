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
            B_matrix[state,neuron] = np.inner(np.multiply(alphas[:-1,state],bettas[:-1,state]),neural_data_matrix.mean(axis=0)[neuron,1:]) / np.inner(alphas[:,state],bettas[:,state])
    B_matrix[B_matrix < 1] = 1.0
    return B_matrix

def update_B_matrix_2(gammas, neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    # alphas and bettas are matrices where axis 0 - time and 1 - states
    time_points = neural_data_matrix.shape[2]
    num_of_states = gammas.shape[1]
    num_of_neurons = neural_data_matrix.shape[1]
    B_matrix = np.zeros((num_of_states, num_of_neurons)) # axis 0 is the state and axis 1 is the neuron
    mean_trialed = neural_data_matrix.mean(axis=0)

    for state in range(num_of_states):
        state_gammas = gammas[:,state] # all gammas across time for this state
        firing_probs = mean_trialed.dot(state_gammas)
        normalized_firing_probs = firing_probs / state_gammas.sum()
        B_matrix[state,:] = normalized_firing_probs

    B_matrix[B_matrix<1] = 1.0
    return B_matrix


def update_B_matrix_3(alphas, bettas, neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    # alphas and bettas are matrices where axis 0 - time and 1 - states
    time_points = neural_data_matrix.shape[2]
    num_of_states = alphas.shape[1]
    num_of_neurons = neural_data_matrix.shape[1]
    B_matrix = np.zeros((num_of_states, num_of_neurons)) # axis 0 is the state and axis 1 is the neuron

    for neuron in range(num_of_neurons):
        for state in range(num_of_states):
            trial_summing_abntk = 0
            trial_summing_ab = 0
            for trial in range(neural_data_matrix.shape[0]):
                summing = 0
                for time in range(time_points-1):
                    B_matrix[state,neuron] += alphas[time,state]*neural_data_matrix[trial,neuron,time+1]*bettas[state,time]
                    summing += alphas[state,time]*bettas[state,time]
            B_matrix[state,neuron] = tr
    return B_matrix