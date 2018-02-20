import numpy as np
from numpy.random import poisson, randint

def create_fake_data_shenoy_model_general(amount_of_trials=72,amount_of_neurons=50):
    # trials, neurons, time
    BL = np.hstack((np.abs(np.random.normal(loc=0.05,scale=0.05,size=(5,int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=0.3,scale=0.05,size=(5,int(amount_of_neurons*0.2))))))
    iden = np.hstack((np.abs(np.random.normal(loc=0.2, scale=0.05, size=(1, int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=0.6, scale=0.075, size=(1, int(amount_of_neurons*0.2))))))
    iden2 = np.hstack((np.abs(np.random.normal(loc=0.35, scale=0.05, size=(1, int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=1.05, scale=0.075, size=(1, int(amount_of_neurons*0.2))))))
    iden3 = np.hstack((np.abs(np.random.normal(loc=0.5, scale=0.05, size=(1, int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=1.5, scale=0.075, size=(1, int(amount_of_neurons*0.2))))))
    iden4 = np.hstack((np.abs(np.random.normal(loc=0.65, scale=0.05, size=(1, int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=1.95, scale=0.075, size=(1, int(amount_of_neurons*0.2))))))
    palat = np.hstack((np.abs(np.random.normal(loc=0.6, scale=0.075, size=(1, int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=1.8, scale=0.125, size=(1, int(amount_of_neurons*0.2))))))
    palat2 = np.hstack((np.abs(np.random.normal(loc=0.9, scale=0.075, size=(1, int(amount_of_neurons*0.8)))),np.abs(np.random.normal(loc=2.7, scale=0.125, size=(1, int(amount_of_neurons*0.2))))))
    neural_rates_per_state = np.concatenate((BL,iden,iden2,iden3,iden4,palat,palat2),axis=0)

    neural_data = np.zeros((amount_of_trials,amount_of_neurons,99),dtype=np.uint8)
    for trial in range(amount_of_trials):

        trial_state_path = np.zeros(99)
        first_state_start = randint(21, 26)
        second_state_start = randint(38, 43)
        # first_state_start = 20
        # second_state_start = 40
        trial_state_path[:first_state_start] = np.random.randint(0,5,size=first_state_start)

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
            neural_data[trial, :, int(time)] = poisson(neural_rates_per_state[int(trial_state_path[int(time)]), :])
        if trial in (0,1,2):
            print(first_state_start,second_state_start,trial_state_path)
    # print("actual neural rates")
    # print(neural_rates_per_state)
    print(neural_data.shape)
    print()
    print(neural_data[0,-1,:])
    print(neural_data[1, -1, :])
    print(neural_data[2, -1, :])

    return neural_data,neural_rates_per_state

def create_fake_data_shenoy_model_specific(amount_of_trials=72,amount_of_neurons=10):
    # trials, neurons, time
    BL = np.ones((4, amount_of_neurons))
    BL1 = np.ones((1, amount_of_neurons))*10
    iden = np.ones((3, amount_of_neurons))
    iden1 = np.ones((1,amount_of_neurons))*50
    palat = np.ones((1, amount_of_neurons))
    palat1 = np.ones((1,amount_of_neurons))*100
    neural_rates_per_state = np.concatenate((BL,BL1,iden,iden1,palat,palat1),axis=0)

    neural_data = np.zeros((amount_of_trials,amount_of_neurons,99))
    for trial in range(amount_of_trials):

        trial_state_path = np.zeros(99)*4
        # first_state_start = randint(21, 26)
        # second_state_start = randint(38, 43)
        first_state_start = 20
        second_state_start = 40

        if trial<18:
            trial_state_path[first_state_start:second_state_start] = 8
            trial_state_path[second_state_start:] = 10
        elif trial<36:
            trial_state_path[first_state_start:second_state_start] = 8
            trial_state_path[second_state_start:] = 10
        elif trial<54:
            trial_state_path[first_state_start:second_state_start] = 8
            trial_state_path[second_state_start:] = 10
        else:
            trial_state_path[first_state_start:second_state_start] = 8
            trial_state_path[second_state_start:] = 10

        for time in range(99):
            # print(time)
            # print(neural_rates_per_state)
            # print(int(trial_state_path[int(time)]))
            # print(neural_rates_per_state[int(trial_state_path[int(time)])])
            neural_data[trial, :, int(time)] = neural_rates_per_state[int(trial_state_path[int(time)])]
    print("actual neural rates")
    print(neural_rates_per_state)
    return neural_data