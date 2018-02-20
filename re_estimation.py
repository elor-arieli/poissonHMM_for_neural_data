import numpy as np


def update_pi_array(gammas):
    gamma_array = gammas.mean(axis=0)[0, :]
    return gamma_array/gamma_array.sum()


def update_Aij_matrix(zettas, gammas):
    num_of_states = gammas.shape[2]
    num_of_trials = gammas.shape[0]
    Aij_matrix = np.zeros((num_of_states,num_of_states))

    for state_i in range(num_of_states):
        for state_j in range(num_of_states):
            zetta_sum = 0
            gamma_sum = 0
            for trial in range(num_of_trials):
                zetta_sum += zettas[trial,:,state_i,state_j].sum()
                gamma_sum += gammas[trial,:,state_i].sum()
            # print("zetta and gamma sums:")
            # print(zetta_sum,gamma_sum)
            Aij_matrix[state_i,state_j] = zetta_sum / gamma_sum
        Aij_matrix[state_i,:] /= Aij_matrix[state_i,:].sum()
    Aij_matrix[np.isnan(Aij_matrix)] = 0
    return Aij_matrix


def update_B_matrix(alphas, bettas, neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    # alphas and bettas are matrices where axis 0 - trials, axis 1 - time and 2 - states
    time_points = neural_data_matrix.shape[2]
    num_of_states = alphas.shape[2]
    num_of_trials = alphas.shape[0]
    num_of_neurons = neural_data_matrix.shape[1]
    B_matrix = np.zeros((num_of_states, num_of_neurons)) # axis 0 is the state and axis 1 is the neuron

    for neuron in range(num_of_neurons):
        for state in range(num_of_states):
            sum_top = 0
            sum_bottom = 0
            for trial in range(num_of_trials):
                sum_top += np.inner(np.multiply(alphas[trial,:-1,state],bettas[trial,:-1,state]),neural_data_matrix[trial,neuron,1:])
                sum_bottom += np.inner(alphas[trial,:,state],bettas[trial,:,state])
            # print("top sum, bottom")
            # print(sum_top,sum_bottom)
            B_matrix[state,neuron] = sum_top / sum_bottom
    B_matrix[np.isnan(B_matrix)] = 0.05
    B_matrix[B_matrix < 0.05] = 0.05
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