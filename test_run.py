from HMM_runner import HMM_model

model = HMM_model()
# model.load_model_params_from_file("D:\\Users\\AnanM4\\PycharmProjects\\HMM_for_neural_data\\test_model_V5_same_trials.pkl")
# print("model pi array:")
# print(model.model_params["pi array"])
# print("model Aij matrix:")
# print(model.model_params["Aij matrix"])
# print("model B matrix:")
# print(model.model_params["B matrix"])
# for key in model.estimation_params.keys():
#     print("model {}".format(key))
#     print(model.estimation_params[key])

model.run_hmm_multi_trial(amount_of_trials=1)
model.save_model_to_file("D:\\Users\\AnanM4\\PycharmProjects\\HMM_for_neural_data\\test_model_V6_same_trials_betta normalized with alphas and path with gammas")
print("model B matrix:")
print(model.model_params["B matrix"])

# all_trials = model.multi_trial_tester()
#
all_trials = [model.single_trial_tester(i) for i in range(5)]
i = 0
for tri in all_trials:

    i+=1
    print("trial #{}".format(i))
    print("path probablity = {}".format(tri[0]))
    print("state path = {}".format(tri[1]))
    print("**********************************************")