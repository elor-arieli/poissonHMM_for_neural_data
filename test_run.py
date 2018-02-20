from HMM_runner import HMM_model
from time import time,strftime,gmtime
# model = HMM_model(["EA54 pre cta only taste responsive neuron dic.pkl"])
model = HMM_model()
# model.load_model_params_from_file("D:\\Users\\AnanM4\\PycharmProjects\\HMM_for_neural_data\\real_data_V1_EA56_PRE.pkl")
# print("model pi array:")
# print(model.model_params["pi array"])
# print("model Aij matrix:")
# print(model.model_params["Aij matrix"])
# print("model B matrix:")
# print(model.model_params["B matrix"])
# for key in model.estimation_params.keys():
#     print("model {}".format(key))
#     print(model.estimation_params[key])
print("starting timer")
t0 = time()
model.run_hmm_multi_trial(amount_of_trials=20)
total_run_time = strftime('%H:%M:%S', gmtime(time()-t0))
# print("total rum time = {0:02f} minutes".format((time()-t0)/60.0))
print("total rum time = {}".format(total_run_time))
model.save_model_to_file("D:\\Users\\AnanM4\\PycharmProjects\\HMM_for_neural_data\\fake_more_real_maybe")
# print("model B matrix:")
# print(model.model_params["B matrix"])

# all_trials = model.multi_trial_tester()
#
all_trials = [model.single_trial_tester(i) for i in range(72)]
i = 0
for tri in all_trials:

    i+=1
    print("trial #{}".format(i))
    print("path probablity = {}".format(tri[0]))
    print("state path = {}".format(tri[1]))
    print("**********************************************")