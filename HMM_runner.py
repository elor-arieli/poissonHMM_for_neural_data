from params_initialization import create_pi_array,create_transition_matrix,create_B_matrix_poissonian_rates
from get_data_from_files import get_and_merge_data_from_pickle_files_list
from calc_params import *
from re_estimation import *
import pickle

import os

class HMM_model(object):
    def __init__(self,file_list, directory=None, num_of_states=11, pi_array_method= "first five BL",
                transition_matrix_type="shenoy", start_time_of_trials=1200, amount_of_trials_per_taste=18,
                start_time=-1, stop_time=4, bin_size=0.05, taste_list=('water','sugar','nacl','CA')):

        if not directory:
            directory = os.getcwd()
        print("Initiating HMM model")
        print("Getting data from files")
        self.neural_data_matrix = get_and_merge_data_from_pickle_files_list(directory,file_list,start_time_of_trials,
                                                                            amount_of_trials_per_taste,start_time,
                                                                            stop_time,bin_size,taste_list)

        print("Initiating base model params")
        self.num_of_states = num_of_states
        self.pi_array_method = pi_array_method
        self.transition_matrix_type = transition_matrix_type
        self.initialize_model_params()


        print("calculating first estimation")
        self.update_estimation_params()
        print("Model Initialized - but not yet optimized")


    def run_hmm_multi_trial(self, amount_of_trials=100, epsilon=10**-4):
        self.optimal_models = []
        for trial_num in range(1,amount_of_trials+1):
            print("Running trial {} for local maxima identification".format(trial_num))
            self.initialize_model_params()
            self.update_estimation_params()
            self.optimize_params(epsilon)

        self.optimal_models.sort(key=lambda x: x[0],reverse=True)
        best_run = self.optimal_models[0]
        self.best_fitting_model = best_run[2] # 0 - path probability, 1 - the state path, 2 - the model param dic
        print("finished optimizing for {} local maxima and found the best path probability is: {}".format(amount_of_trials,best_run[0]))


    def initialize_model_params(self):
        self.model_params = {"pi array": create_pi_array(self.num_of_states, self.pi_array_method),
                             "Aij matrix": create_transition_matrix(self.num_of_states, self.transition_matrix_type),
                             "B matrix": create_B_matrix_poissonian_rates(self.neural_data_matrix)}


    def optimize_params(self,epsilon=10**-4):
        print("Initiating model parameter optimization")
        last_path_prob = 0

        trial_num = 0
        while np.abs(self.estimation_params["path prob"] - last_path_prob) > epsilon:
            trial_num += 1
            print("Running Estimation-Maximization trial #{}".format(trial_num))
            self.update_model_params()
            self.update_estimation_params()

        self.optimal_models.append((self.estimation_params["path prob"],self.estimation_params["best path"],self.model_params))


    def update_model_params(self):
        self.model_params["pi array"] = update_pi_array(self.estimation_params["gamma"])
        self.model_params["Aij matrix"] = update_Aij_matrix(self.estimation_params["zetta"],
                                                            self.estimation_params["gamma"])
        self.model_params["B matrix"] = update_B_matrix(self.estimation_params["alpha"],
                                                        self.estimation_params["betta"],
                                                        self.neural_data_matrix)

    def update_estimation_params(self):
        self.estimation_params = {"alpha": calc_alphas(self.model_params["pi array"], self.model_params["Aij matrix"],
                                                       self.model_params["B matrix"], self.neural_data_matrix),
                                  "betta": calc_bettas(self.model_params["Aij matrix"],
                                                       self.model_params["B matrix"], self.neural_data_matrix)}

        self.estimation_params["gamma"] = calc_gammas(self.estimation_params["alpha"], self.estimation_params["betta"])
        lamdas, psi_array, path_prob, last_state, path = calc_lamdas_psi(self.model_params["pi array"],
                                                                         self.model_params["Aij matrix"],
                                                                         self.model_params["B matrix"],
                                                                         self.neural_data_matrix)
        self.estimation_params["lambda"] = lamdas
        self.estimation_params["psi"] = psi_array
        self.estimation_params["path prob"] = path_prob
        self.estimation_params["best path"] = path
        self.estimation_params["zetta"] = calc_zettas(self.estimation_params["alpha"], self.estimation_params["betta"],
                                                      self.model_params["Aij matrix"],
                                                      self.model_params["B matrix"], self.neural_data_matrix)


    def save_model_to_file(self, filename):
        with open(filename+".pkl", 'wb') as handle:
            pickle.dump(self.best_fitting_model, handle, protocol=pickle.HIGHEST_PROTOCOL)