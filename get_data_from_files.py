import numpy as np
import pickle
import os


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def create_trials_matrix(all_neurons_spike_times, event_dic, start_time=-1, stop_time=4, bin_size=0.05, taste_list=('water','sugar','nacl','CA'),bad_elec_list=[]):
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
                    spikes_in_bin = hist1 / bin_size
                    all_neural_responses_for_event.append(spikes_in_bin)
            matrix_dic.append(all_neural_responses_for_event)
    matrix_dic = np.array(matrix_dic)
    # print(taste_event_amount)
    return matrix_dic

def get_data_from_pickle_files_directory(directory,start_time_of_trials=1200, amount_of_trials_per_taste=20,start_time=-1, stop_time=4, bin_size=0.05, taste_list=('water','sugar','nacl','CA')):
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