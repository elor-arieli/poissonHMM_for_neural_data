import numpy as np
from auxilary_functions import poisson_prob_population_vec

def calc_alphas(pi_array,Aij_matrix,B_matrix,neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    time_points = neural_data_matrix.shape[2]
    num_of_states = B_matrix.shape[0]
    alphas = np.zeros((time_points,num_of_states)) # axis 0 is times, axis 1 is states
    alphas[0,:] = pi_array

    for state in range(num_of_states):
        alphas[0,state] *= poisson_prob_population_vec(B_matrix[state,:],neural_data_matrix[:,:,0])

    for time in range(1,time_points):
        for state_j in range(num_of_states):
            sum_transition_probs = 0

            for state_i in range(num_of_states):
                sum_transition_probs += alphas[time-1,state_i]*Aij_matrix[state_i,state_j]

            alphas[time,state_j] = sum_transition_probs*poisson_prob_population_vec(B_matrix[state_j,:],neural_data_matrix[:,:,time])
    return alphas

def calc_bettas(Aij_matrix,B_matrix,neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    time_points = neural_data_matrix.shape[2]
    num_of_states = B_matrix.shape[0]
    bettas = np.zeros((time_points, num_of_states))  # axis 0 is times, axis 1 is states
    bettas[-1, :] = np.ones(bettas.shape[1])

    for time in range(time_points-2,-1,-1):
        for state_i in range(num_of_states):
            sum_transition_probs = 0

            for state_j in range(num_of_states):
                sum_transition_probs += bettas[time+1, state_j] * \
                                        Aij_matrix[state_i, state_j] * \
                                        poisson_prob_population_vec(B_matrix[state_j, :],neural_data_matrix[:, :, time+1])

            bettas[time, state_i] = sum_transition_probs

    return bettas

def calc_gammas(alphas, bettas):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    time_points = alphas.shape[0]
    num_of_states = alphas.shape[1]
    gammas = np.zeros((time_points, num_of_states))  # axis 0 is times, axis 1 is states

    for time in range(time_points):
        summation = np.inner(alphas[time,:],bettas[time,:])

        for state_i in range(num_of_states):
            gammas[time,state_i] = bettas[time,state_i]*alphas[time,state_i]/summation

    return gammas

def calc_lamdas_psi(pi_array,Aij_matrix,B_matrix,neural_data_matrix):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    time_points = neural_data_matrix.shape[2]
    num_of_states = B_matrix.shape[0]
    lamdas = np.zeros((time_points, num_of_states))  # axis 0 is times, axis 1 is states
    psi_array = np.zeros((time_points, num_of_states))

    for state in range(num_of_states):
        lamdas[0, state] = pi_array[state]*poisson_prob_population_vec(B_matrix[state, :],neural_data_matrix[:, :, 0])

    for time in range(1,time_points):
        for state_j in range(num_of_states):
            previous_lamdas_prob_array = np.zeros(num_of_states)

            for state_i in range(num_of_states):
                previous_lamdas_prob_array[state_i] = lamdas[time-1,state_i] * Aij_matrix[state_i,state_j]

            psi_array[time,state_j] = previous_lamdas_prob_array.argmax()
            lamdas[time,state_j] = previous_lamdas_prob_array.max() * \
                                   poisson_prob_population_vec(B_matrix[state_j,:],neural_data_matrix[:,:,time])

    path_prob = lamdas[-1,:].max()
    last_state = lamdas[-1,:].argmax()
    path = np.zeros(time_points)
    path[-1] = last_state
    for time in range(time_points-2,-1,-1):
        path[time] = psi_array[time+1,path[time+1]]

    return lamdas,psi_array,path_prob,last_state,path