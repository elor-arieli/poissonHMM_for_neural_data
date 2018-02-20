import numpy as np
import pickle
import os


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def create_trials_matrix(all_neurons_spike_times, event_dic, start_time=-1, stop_time=4, bin_size=0.01, taste_list=('water','sugar','nacl','CA'), change_to_fire_rate=False, bad_elec_list=[]):
    matrix_dic = []
    # taste_event_amount = {'water': 0, 'sugar': 0, 'nacl': 0, 'CA': 0}
    bin_amount = (stop_time-start_time)//bin_size
    for taste in taste_list:
        for event in event_dic[taste]:
            all_neural_responses_for_event = []
            # taste_event_amount[taste] += 1
            for neural_spike_times in all_neurons_spike_times:
                if neural_spike_times[0] not in bad_elec_list:
                    spikes = [neural_spike_times[2][i] - event for i in range(len(neural_spike_times[2])) if start_time < neural_spike_times[2][i] - event < stop_time]
                    hist1, bin_edges = np.histogram(spikes, int(bin_amount), (start_time, stop_time))
                    if change_to_fire_rate == "rate":
                        hist1 = hist1 / bin_size
                    all_neural_responses_for_event.append(hist1)
            matrix_dic.append(all_neural_responses_for_event)
    matrix_dic = np.array(matrix_dic)
    # print(taste_event_amount)
    return matrix_dic

def get_data_from_pickle_files_directory(directory,start_time_of_trials=1200, amount_of_trials_per_taste=18,start_time=-1, stop_time=4, bin_size=0.05, taste_list=('water','sugar','nacl','CA')):
    matrices = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            with (open(filename, "rb")) as openfile:
                while True:
                    try:
                        dic = pickle.load(openfile)
                    except EOFError:
                        break
            for taste in dic['event_times'].keys():
                dic['event_times'][taste] = [i for i in dic['event_times'][taste] if i>start_time_of_trials][:amount_of_trials_per_taste]
            matrices[filename] = create_trials_matrix(dic['neurons'], dic['event_times'],start_time, stop_time, bin_size, taste_list)
    return matrices


def get_and_merge_data_from_pickle_files_list(directory, file_list, start_time_of_trials=1200, amount_of_trials_per_taste=18,start_time=-1, stop_time=4, bin_size=0.01, taste_list=('water','sugar','nacl','CA')):
    matrice_list = []
    for filename in file_list:
        with (open(directory + "\\" + filename, "rb")) as openfile:
            while True:
                try:
                    dic = pickle.load(openfile)
                except EOFError:
                    break
        for taste in taste_list:
            dic['event_times'][taste] = [i for i in dic['event_times'][taste] if i>start_time_of_trials][:amount_of_trials_per_taste]
        matrice_list.append(create_trials_matrix(dic['neurons'], dic['event_times'],start_time, stop_time, bin_size, taste_list))
        amount_of_neurons = matrice_list[-1].shape[1]
        bad_neurons = []
        for neuron in range(amount_of_neurons):
            if matrice_list[-1][:,neuron,:].mean() < 0.25:
                bad_neurons.append(neuron)
            elif np.abs(matrice_list[-1][:,neuron,0:20].mean() - matrice_list[-1][:,neuron,30:80].mean()) < 0.1:
                bad_neurons.append(neuron)
            # elif np.abs(matrice_list[-1][:, neuron, :20].mean() - matrice_list[-1][:, neuron, 30:80].mean()) < 0.1:
            #     bad_neurons.append(neuron)
        matrice_list[-1] = np.delete(matrice_list[-1],tuple(bad_neurons),axis=1)
    if len(matrice_list) == 1:
        full_matrix = matrice_list[0]
    else:
        full_matrix = np.concatenate(matrice_list, axis=1)
    print(full_matrix.shape)
    return full_matrix
