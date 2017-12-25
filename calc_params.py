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
        for entering_state in range(num_of_states):
            sum_transition_probs = 0

            for leaving_state in range(num_of_states):
                sum_transition_probs += alphas[time,leaving_state]*Aij_matrix[leaving_state,entering_state]

            alphas[time,state] = sum_transition_probs*poisson_prob_population_vec(B_matrix[state,:],neural_data_matrix[:,:,time])

def calc_bettas():
    pass

def calc_gammas():
    pass