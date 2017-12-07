import numpy as np
import random


def create_transition_matrix(num_of_states, type='FX'):
    transition_matrix = np.zeros((num_of_states,num_of_states))
    diag = random.uniform(0.9, 0.95)
    if type == "FX":
        for i in range(num_of_states):
            for j in range(num_of_states):
                if i == j:
                    transition_matrix[i,j] = diag
                elif i == j-1:
                    transition_matrix[i, j] = 1-diag
    elif type == "F":
        for i in range(num_of_states):
            for j in range(num_of_states):
                if i == j:
                    transition_matrix[i,j] = diag
                elif i < j:
                    transition_matrix[i, j] = (1 - diag) / sum(np.arange(num_of_states) - i)*(num_of_states - j + 1)
    elif type == "ATA":
        for i in range(num_of_states):
            for j in range(num_of_states):
                if i == j:
                    transition_matrix[i, j] = diag
                elif i < j:
                    transition_matrix[i, j] = (1 - diag) / (num_of_states - 1)
    else:
        raise SyntaxError("type is not valid, should be F, FX or ATA")

def create_B_matrix_poissonian_rates(num_of_states,num_of_neurons?, neural data matrix?):
    pass


def create_pi_array(num_of_states,start_from="first"):
    pi_array = np.zeros(num_of_states)
    if start_from=="first":
        pi_array[0] = 1
    else:
        pi_array += (1.0/num_of_states)
    return pi_array