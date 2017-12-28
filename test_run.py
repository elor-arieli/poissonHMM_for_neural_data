from HMM_runner import HMM_model

model = HMM_model()

model.run_hmm_multi_trial(amount_of_trials=20)
model.save_model_to_file("D:\\Users\\AnanM4\\PycharmProjects\\HMM_for_neural_data\\test_model_20_runs_new_convergance")

all_trials = model.multi_trial_tester()

i = 0
for tri in all_trials:
    i+=1
    print("trial #{}".format(i))
    print("path probablity = {}".format(tri[0]))
    print("state path = {}".format(tri[1]))
    print("**********************************************")