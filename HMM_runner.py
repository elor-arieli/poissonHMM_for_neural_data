from params_initialization import create_pi_array,create_transition_matrix,create_B_matrix_poissonian_rates
from get_data_from_files import get_and_merge_data_from_pickle_files_list
from calc_params import *
from re_estimation import *
from test_data_creator import *
import pickle
import os
from tqdm import tqdm
import sys

class HMM_model(object):
    def __init__(self,file_list=None, directory=None, num_of_states=11, pi_array_method= "first five BL",
                transition_matrix_type="shenoy", start_time_of_trials=1200, amount_of_trials_per_taste=18,
                start_time=-1, stop_time=4, bin_size=0.05, taste_list=('water','sugar','nacl','CA')):

        if not directory:
            directory = os.getcwd()
        print("Initiating HMM model")
        print("Getting data from files")
        if file_list:
            self.neural_data_matrix = get_and_merge_data_from_pickle_files_list(directory,file_list,start_time_of_trials,
                                                                            amount_of_trials_per_taste,start_time,
                                                                            stop_time,bin_size,taste_list)
            self.fake_b_matrix = None
        else:
            self.neural_data_matrix, self.fake_b_matrix = create_fake_data_shenoy_model_general()
            print(self.fake_b_matrix)

        print("Initiating base model params")
        self.num_of_states = num_of_states
        self.pi_array_method = pi_array_method
        self.transition_matrix_type = transition_matrix_type
        self.initialize_model_params()


        # print("calculating first estimation")
        # self.update_estimation_params()
        print("Model Initialized - but not yet optimized")

    def run_hmm_multi_trial(self, amount_of_trials=100, epsilon=0.01):
        self.optimal_models = []
        for trial_num in range(1,amount_of_trials+1):
            print("Running trial {} for local maxima identification".format(trial_num))
            self.initialize_model_params()
            # self.update_estimation_params()
            self.optimize_params(epsilon)
            all_trials = [self.single_trial_tester(i) for i in [1,10,20,30,40,50,60,70]]
            print("2Wat, 2Suc, 2Na, 2CA")
            for tri in all_trials:
                print("path probablity = {}".format(tri[0]))
                print("state path = {}".format(tri[1]))
                print("**********************************************")

        self.optimal_models.sort(key=lambda x: x[0] if np.isnan(x[0])==False else 0,reverse=True)
        best_run = self.optimal_models[0]
        self.best_fitting_model = best_run[2] # 0 - path probability, 1 - the state path, 2 - the model param dic
        print("finished optimizing for {} local maxima and found the best path probability is: {}".format(amount_of_trials,best_run[0]))


    def initialize_model_params(self):
        self.model_params = {"pi array": create_pi_array(self.num_of_states, self.pi_array_method),
                             "Aij matrix": create_transition_matrix(self.num_of_states, self.transition_matrix_type),
                             "B matrix": create_B_matrix_poissonian_rates(self.neural_data_matrix)}


    def optimize_params(self,epsilon=10**-2):
        print("Initiating model parameter optimization")
        print("calculating first estimation")
        self.update_estimation_params()
        self.optimal_models.append((self.estimation_params["path prob"], self.estimation_params["best path"], self.model_params))
        last_path_prob = 1
        # print((np.log(self.estimation_params["path prob"]) - last_path_prob) / last_path_prob)
        new_path_prob = self.calc_model_convergance()
        trial_num = 0
        # while np.abs((np.log(self.estimation_params["path prob"]) - last_path_prob)/last_path_prob) > epsilon:
        while trial_num < 10 and np.abs((new_path_prob - last_path_prob) / last_path_prob) > epsilon:
            trial_num += 1
            print("Running Estimation-Maximization trial #{}".format(trial_num))
            # print("diff model B_matrix - real B_matrix = {}".format(self.model_params["B matrix"]-self.fake_b_matrix))
            # print("sum of differences = {}".format((np.abs(self.model_params["B matrix"] - self.fake_b_matrix)).sum()))
            # print("Improvement from last time = {}".format(np.abs((np.log(self.estimation_params["path prob"]) - last_path_prob)/last_path_prob)))
            print("Last LL = {}, current LL = {}, Improvement from last time = {}".format(last_path_prob,new_path_prob,np.abs((new_path_prob - last_path_prob) / last_path_prob)))
            if self.fake_b_matrix is not None:
                print("sum of distances from model B matrix to actual B matix = {}".format(np.abs(self.model_params["B matrix"] - self.fake_b_matrix).sum()))
                # print("fake_b_matrix neuron 1: ", self.fake_b_matrix[:, 0])
                # print("B matrix neuron 1: ", self.model_params["B matrix"][:,0])
                # print("pi array sum: ", self.model_params["pi array"].sum())
                # print("Aij matrix sum: ", self.model_params["Aij matrix"].sum())
            # last_path_prob = np.log(self.estimation_params["path prob"])
            self.update_model_params()
            self.update_estimation_params()
            last_path_prob, new_path_prob = new_path_prob, self.calc_model_convergance()
            if np.isnan(new_path_prob):
                print("reached NaN value in convergence - breaking!")
                break
        # print("Last Improvement = {}".format(np.abs((np.log(self.estimation_params["path prob"]) - last_path_prob) / last_path_prob)))
        print("Last Improvement = {}".format(np.abs((new_path_prob - last_path_prob) / last_path_prob)))
        self.optimal_models.append((self.estimation_params["path prob"],self.estimation_params["best path"],self.model_params))


    def calc_model_convergance(self):
        mat = self.estimation_params["C_array"]
        mat[mat == 0] = 1
        log_mat = np.log(mat)
        # print(log_mat)
        log_mat[log_mat<-1E300] = 0
        log_mat[np.isnan(log_mat)] = 0
        # print(log_mat.sum(),log_mat)
        return -1*log_mat.sum()


    def update_model_params(self):
        self.model_params["pi array"] = update_pi_array(self.estimation_params["gamma"])
        self.model_params["Aij matrix"] = update_Aij_matrix(self.estimation_params["zetta"],
                                                            self.estimation_params["gamma"])
        self.model_params["B matrix"] = update_B_matrix(self.estimation_params["alpha"],
                                                        self.estimation_params["betta"],
                                                        self.neural_data_matrix)

        # self.model_params["B matrix"] = update_B_matrix_2(self.estimation_params["gamma"],self.neural_data_matrix)

    def update_estimation_params(self,trial_num=False):
        if trial_num:
            self.estimation_params = {}
            self.estimation_params["alpha"],self.estimation_params["C_array"] = calc_alphas(self.model_params["pi array"], self.model_params["Aij matrix"],
                                                           self.model_params["B matrix"], self.neural_data_matrix,trial_num=trial_num)
            self.estimation_params["betta"] = calc_bettas(self.model_params["Aij matrix"],self.model_params["B matrix"],
                                                           self.estimation_params["alpha"],self.estimation_params["C_array"],self.neural_data_matrix,trial_num=trial_num)

            self.estimation_params["gamma"] = calc_gammas(self.estimation_params["alpha"], self.estimation_params["betta"])
            # lamdas, psi_array, path_prob, last_state, path = calc_lamdas_psi(self.model_params["pi array"],
            #                                                                  self.model_params["Aij matrix"],
            #                                                                  self.model_params["B matrix"],
            #                                                                  self.neural_data_matrix,trial_num=trial_num)
            # self.estimation_params["lambda"] = lamdas
            # self.estimation_params["psi"] = psi_array
            self.estimation_params["path prob"] = self.calc_model_convergance()
            self.estimation_params["best path"] = self.estimation_params["gamma"][0].argmax(axis=1)
            self.estimation_params["zetta"] = calc_zettas(self.estimation_params["alpha"], self.estimation_params["betta"],
                                                          self.model_params["Aij matrix"],
                                                          self.model_params["B matrix"], self.neural_data_matrix,trial_num=trial_num)
        else:
            self.estimation_params = {}
            self.estimation_params["alpha"],self.estimation_params["C_array"] = calc_alphas(self.model_params["pi array"], self.model_params["Aij matrix"],
                                     self.model_params["B matrix"], self.neural_data_matrix)
            self.estimation_params["betta"] = calc_bettas(self.model_params["Aij matrix"],self.model_params["B matrix"],
                                     self.estimation_params["alpha"],self.estimation_params["C_array"],self.neural_data_matrix)

            self.estimation_params["gamma"] = calc_gammas(self.estimation_params["alpha"],
                                                          self.estimation_params["betta"])
            # lamdas, psi_array, path_prob, last_state, path = calc_lamdas_psi(self.model_params["pi array"],
            #                                                                  self.model_params["Aij matrix"],
            #                                                                  self.model_params["B matrix"],
            #                                                                  self.neural_data_matrix)
            # self.estimation_params["lambda"] = lamdas
            # self.estimation_params["psi"] = psi_array
            self.estimation_params["path prob"] = self.calc_model_convergance()
            # self.estimation_params["best path"] = path
            self.estimation_params["best path"] = self.estimation_params["gamma"].argmax(axis=2)
            self.estimation_params["zetta"] = calc_zettas(self.estimation_params["alpha"],
                                                          self.estimation_params["betta"],
                                                          self.model_params["Aij matrix"],
                                                          self.model_params["B matrix"], self.neural_data_matrix)


    def save_model_to_file(self, filename):
        with open(filename+".pkl", 'wb') as handle:
            pickle.dump(self.best_fitting_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(filename + "_all_models.pkl", 'wb') as handle:
            pickle.dump({"all optimized models": self.optimal_models}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model_params_from_file(self,filename):
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    dic = pickle.load(openfile)
                except EOFError:
                    break
        self.model_params = dic

    def single_trial_tester(self,trial_num):
        print("running path identification of trial {}".format(trial_num))
        self.update_estimation_params(trial_num=trial_num)
        return (self.estimation_params["path prob"],self.estimation_params["gamma"][0].argmax(axis=1))

    def multi_trial_tester(self):
        all_trials = []
        for trial in range(self.neural_data_matrix.shape[0]):
            print("checking state path in trial {}".format(trial))
            all_trials.append(self.single_trial_tester(trial))
        return all_trials
