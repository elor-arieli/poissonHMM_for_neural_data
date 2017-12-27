import numpy as np
from numpy.random import poisson, randint

def create_fake_data_shenoy_model(amount_of_trials=72,amount_of_neurons=10):
    # trials, neurons, time
    BL = np.abs(np.random.normal(loc=2.0,scale=3.0,size=(5,amount_of_neurons)))
    iden = np.abs(np.random.normal(loc=5.0, scale=2.0, size=(4, amount_of_neurons)))
    palat = np.abs(np.random.normal(loc=8.0, scale=2.5, size=(2, amount_of_neurons)))
    neural_rates_per_state = np.concatenate((BL,iden,palat),axis=0)

    neural_data = np.zeros((amount_of_trials,amount_of_neurons,99))
    for trial in range(amount_of_trials):

        trial_state_path = np.array([0,0,0,0,0,1,1,1,1,1,3,3,3,3,4,4,4,4,2,2,2,2] + [0 for i in range(77)])
        first_state_start = randint(21, 26)
        second_state_start = randint(38, 43)

        if trial<18:
            trial_state_path[first_state_start:second_state_start] = 5
            trial_state_path[second_state_start:] = 9
        elif trial<36:
            trial_state_path[first_state_start:second_state_start] = 6
            trial_state_path[second_state_start:] = 9
        elif trial<54:
            trial_state_path[first_state_start:second_state_start] = 7
            trial_state_path[second_state_start:] = 9
        else:
            trial_state_path[first_state_start:second_state_start] = 8
            trial_state_path[second_state_start:] = 10

        for time in range(99):
            # print(poisson(neural_rates_per_state[trial_state_path[time],:]))
            neural_data[trial,:,time] = poisson(neural_rates_per_state[trial_state_path[time],:])

    return neural_data