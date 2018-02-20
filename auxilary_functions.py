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
            # print(prob_array.mean(),prob_array)
            # print(prob)
    else:
        for neuron in range(spike_count_matrix.shape[0]):
            prob *= poisson_prob(firing_rate_vec[neuron],spike_count_matrix[neuron])
    return prob

def poisson_prob(lamda, k): # lamda = fire rate in this bin, k = amount of spikes
    # if ((lamda**k)*np.exp(-1*lamda))/factorial(k) < 0:
    #     print("lamda: {}, K: {}, result: {}".format(lamda,k,((lamda**k)*np.exp(-1*lamda))/factorial(k)))
    # try:
    #     prob = ((lamda**k)*np.exp(-1*lamda))/factorial(k)
    # except:
    #     print(lamda,k)
    #     return 0
    # if np.isnan(prob):
    #     print(prob)
    # return prob
    return ((lamda**k)*np.exp(-1*lamda))/factorial(k)

def range_prod(lo,hi):
    if lo+1 < hi:
        mid = (hi+lo)//2
        return range_prod(lo,mid) * range_prod(mid+1,hi)
    if lo == hi:
        return lo
    return lo*hi

def treefactorial(n):
    if n < 2:
        return 1
    return range_prod(1,n)