from HMM_runner import HMM_model

model = HMM_model()

model.run_hmm_multi_trial(amount_of_trials=3)

all_trials = model.multi_trial_tester()

i = 0
for tri in all_trials:
    i+=1
    print("trial #{}".format(i))
    print("path probablity = {}".format(all_trials[0]))
    print("state path = {}".format(all_trials[1]))
    print("**********************************************")