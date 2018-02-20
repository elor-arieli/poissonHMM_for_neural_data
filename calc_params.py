import numpy as np
from auxilary_functions import poisson_prob_population_vec
from tqdm import tqdm

def calc_alphas(pi_array,Aij_matrix,B_matrix,neural_data_matrix,trial_num=False):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    time_points = neural_data_matrix.shape[2]
    num_of_states = B_matrix.shape[0]
    if trial_num:
        num_of_trials = 1
    else:
        num_of_trials = neural_data_matrix.shape[0]
    alphas = np.zeros((num_of_trials,time_points,num_of_states)) # axis 0 is trials axis 1 is times, axis 2 is states
    alphas[:,0,:] = pi_array
    C_array = np.zeros((num_of_trials,time_points))

    for state in range(num_of_states):
        for trial in range(num_of_trials):
            if trial_num:
                alphas[trial,0, state] *= poisson_prob_population_vec(B_matrix[state,:],neural_data_matrix[trial_num,:,0])
            else:
                alphas[trial,0, state] *= poisson_prob_population_vec(B_matrix[state, :], neural_data_matrix[trial,:, 0])
    for trial in tqdm(range(num_of_trials),desc="calculating alphas"):
        for time in range(1,time_points):
            C_array[trial,time-1] = alphas[trial,time-1,:].sum()
            alphas[trial,time-1,:] /= alphas[trial,time-1,:].sum() # normalize last stage to sum to 1
            for state_i in range(num_of_states):
                sum_transition_probs = 0

                for state_j in range(num_of_states):

                        # print("****************************************")
                        # print("time: {}, state i: {}, state j: {}".format(time,state_i,state_j))
                        # print(alphas[time - 1, state_j])
                        # print(Aij_matrix[state_i, state_j])
                        # print(poisson_prob_population_vec(B_matrix[state_i, :],neural_data_matrix[:, :, time]))
                    sum_transition_probs += alphas[trial,time - 1, state_j] * Aij_matrix[state_j, state_i]

                if trial_num:
                    alphas[trial, time, state_i] = sum_transition_probs * poisson_prob_population_vec(B_matrix[state_i, :],
                                                                                                   neural_data_matrix[trial_num, :, time])
                else:
                    alphas[trial, time, state_i] = sum_transition_probs * poisson_prob_population_vec(B_matrix[state_i, :],
                                                                                               neural_data_matrix[trial,:, time])
        C_array[trial,-1] = alphas[trial,-1, :].sum()
        alphas[trial,-1, :] /= alphas[trial,-1, :].sum()  # normalize last stage to sum to 1
        alphas[np.isnan(alphas)] = 0
        C_array[np.isnan(C_array)] = 0
    return alphas,C_array


def calc_bettas(Aij_matrix,B_matrix,alphas,C_array, neural_data_matrix,trial_num=False):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    time_points = neural_data_matrix.shape[2]
    num_of_states = B_matrix.shape[0]
    if trial_num:
        num_of_trials = 1
    else:
        num_of_trials = neural_data_matrix.shape[0]
    bettas = np.zeros((num_of_trials,time_points, num_of_states))  # axis 0 is trials, axis 1 is times, axis2 = states
    bettas[:,-1, :] = 1.0

    for trial in tqdm(range(num_of_trials),desc="calculating bettas"):
        for time in range(time_points-2,-1,-1):
            for state_i in range(num_of_states):
                sum_transition_probs = 0

                for state_j in range(num_of_states):
                    if trial_num:
                        sum_transition_probs += bettas[trial,time+1, state_j] * \
                                                Aij_matrix[state_i, state_j] * \
                                                poisson_prob_population_vec(B_matrix[state_j, :],neural_data_matrix[trial_num, :, time+1])
                    else:
                        sum_transition_probs += bettas[trial,time + 1, state_j] * \
                                                Aij_matrix[state_i, state_j] * \
                                                poisson_prob_population_vec(B_matrix[state_j, :],neural_data_matrix[trial,:, time + 1])


                if np.isnan(sum_transition_probs):
                    print()
                bettas[trial,time, state_i] = sum_transition_probs

            # print(alphas[time, :])
            # print(C_array[trial,time])
            # print(bettas[trial,time, :])
            # print(bettas[trial,time, :] / C_array[trial,time])
            bettas[trial,time, :] /= C_array[trial,time]
    bettas[np.isnan(bettas)] = 0
    return bettas


def calc_gammas(alphas, bettas):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    num_of_trials,time_points,num_of_states = alphas.shape
    gammas = np.zeros((num_of_trials,time_points, num_of_states))  # axis 0 is times, axis 1 is states
    for trial in tqdm(range(num_of_trials),desc="calculating gammas"):
        for time in range(time_points):
            # summation = np.inner(alphas[trial,time,:],bettas[trial,time,:])

            for state_i in range(num_of_states):
                # gammas[trial,time,state_i] = bettas[trial,time,state_i]*alphas[trial,time,state_i]/summation
                gammas[trial, time, state_i] = bettas[trial, time, state_i] * alphas[trial, time, state_i]
    gammas[np.isnan(gammas)] = 0
    return gammas


def calc_lamdas_psi(pi_array,Aij_matrix,B_matrix,neural_data_matrix,multi_trial=True):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points

    if multi_trial:
        time_points = neural_data_matrix.shape[2]
    else:
        time_points = neural_data_matrix.shape[1]
    num_of_states = B_matrix.shape[0]
    lamdas = np.zeros((time_points, num_of_states))  # axis 0 is times, axis 1 is states
    psi_array = np.zeros((time_points, num_of_states))

    for state in range(num_of_states):
        if multi_trial:
            lamdas[0, state] = pi_array[state]*poisson_prob_population_vec(B_matrix[state, :],neural_data_matrix[:, :, 0])
        else:
            lamdas[0, state] = pi_array[state] * poisson_prob_population_vec(B_matrix[state, :],neural_data_matrix[:, 0])

    for time in tqdm(range(1,time_points),desc="calculating lamdas and psi"):
        lamdas[time - 1, :] /= lamdas[time - 1, :].sum()  # normalize last stage to sum to 1
        for state_j in range(num_of_states):
            previous_lamdas_prob_array = np.zeros(num_of_states)

            for state_i in range(num_of_states):
                previous_lamdas_prob_array[state_i] = lamdas[time-1,state_i] * Aij_matrix[state_i,state_j]

            psi_array[time,state_j] = previous_lamdas_prob_array.argmax()
            if multi_trial:
                lamdas[time,state_j] = previous_lamdas_prob_array.max() * \
                                       poisson_prob_population_vec(B_matrix[state_j,:],neural_data_matrix[:,:,time])
            else:
                lamdas[time, state_j] = previous_lamdas_prob_array.max() * \
                                        poisson_prob_population_vec(B_matrix[state_j, :],neural_data_matrix[:, time])

    lamdas[-1, :] /= lamdas[-1, :].sum()  # normalize last stage to sum to 1
    path_prob = lamdas[-1,:].max()
    last_state = lamdas[-1,:].argmax()
    path = np.zeros(time_points)
    path[-1] = last_state
    for time in range(time_points-2,-1,-1):
        path[time] = psi_array[time+1,int(path[time+1])]

    return lamdas,psi_array,path_prob,last_state,path


def calc_zettas(alphas,bettas,Aij_matrix,B_matrix,neural_data_matrix,trial_num=False):
    # B matrix is a matrix where axis 0 is the state and axis 1 is the neuron
    # Aij matrix is a matrix where axis 0 is the state i from which we are leaving and axis 1 is state j to which we are going.
    # neural data matrix axis are: 0 - trials, 1 - neurons, 2 - time points
    # alphas and bettas are matrices where axis 0 - time and 1 - states

    time_points = neural_data_matrix.shape[2]
    num_of_states = B_matrix.shape[0]
    if trial_num:
        num_of_trials = 1
    else:
        num_of_trials = alphas.shape[0]

    zettas = np.zeros((num_of_trials,time_points-1, num_of_states, num_of_states))  # axis 0 is times, axis 1 is states
    for trial in tqdm(range(num_of_trials),desc="calculating zettas"):
        for time in range(time_points-1):
            for state_i in range(num_of_states):
                for state_j in range(num_of_states):
                    if trial_num:
                        zettas[trial,time,state_i,state_j] = alphas[trial,time,state_i] * Aij_matrix[state_i,state_j] * bettas[trial,time+1,state_j]\
                                                       * poisson_prob_population_vec(B_matrix[state_j,:],neural_data_matrix[trial_num,:,time+1])
                    else:
                        zettas[trial,time, state_i, state_j] = alphas[trial,time, state_i] * Aij_matrix[state_i, state_j] * \
                                                         bettas[trial,time + 1, state_j] * \
                                                         poisson_prob_population_vec(B_matrix[state_j, :],neural_data_matrix[trial,:, time + 1])

            # zettas[trial,time,:,:] = zettas[trial,time,:,:] / zettas[trial,time,:,:].sum()
    zettas[np.isnan(zettas)] = 0
    return zettas
