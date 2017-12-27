import numpy as np
# import numpy.math.factorial as factorial
from scipy.misc import factorial


def poisson_prob_population_vec(firing_rate_vec, spike_count_matrix):
    # spike_count_matrix axis are: 0 - trials, 1 - neurons
    # gets a matrix of trials and neurons and an array of firing rates and calculates the poisson prob to get these
    # neural responses across neurons and trials
    prob = 1
    if len(spike_count_matrix.shape) == 2:
        for neuron in range(spike_count_matrix.shape[1]):
            prob_array = np.zeros(spike_count_matrix.shape[0])
            for trial in range(spike_count_matrix.shape[0]):
                prob_array[trial] = poisson_prob(firing_rate_vec[neuron],spike_count_matrix[trial,neuron])
            prob *= prob_array.mean()
    else:
        for neuron in range(spike_count_matrix.shape[0]):
            prob *= poisson_prob(firing_rate_vec[neuron],spike_count_matrix[neuron])
    return prob

def poisson_prob(lamda, k): # lamda = fire rate in this bin, k = amount of spikes
    return ((lamda**k)*np.exp(-1*lamda))/factorial(k)