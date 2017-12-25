import numpy as np
from math import factorial,log2


def poisson_prob_population_vec(firing_rate_vec, spike_count_matrix):
    # spike_count_matrix axis are: 0 - trials, 1 - neurons
    # gets a matrix of trials and neurons and an array of firing rates and calculates the poisson prob to get these
    # neural responses across neurons and trials
    prob = 1
    for neuron in range(len(spike_count_matrix.shape[1])):
        for trial in range(len(spike_count_matrix.shape[0])):
            prob *= poisson_prob(firing_rate_vec[neuron],spike_count_matrix[trial,neuron])
    return prob

def poisson_prob(lamda, k): # lamda = fire rate in this bin, k = amount of spikes
    return ((lamda**k)*np.exp(-1*lamda))/factorial(k)