(hilserl4) rwr_team5@SRL-LPT-001:~/hil-serl/examples/experiments/basic_task$ python ../../record_success_fail.py --exp_name basic_task --successes_needed 200

(hilserl4) rwr_team5@SRL-LPT-001:~/hil-serl/examples/experiments/basic_task$ python ../../train_reward_classifier.py --exp_name basic_task

(hilserl4) rwr_team5@SRL-LPT-001:~/hil-serl/examples/experiments/basic_task$ python ../../record_demos.py --exp_name basic_task --successes_needed 20

then in two terminals:
(hilserl4) rwr_team5@SRL-LPT-001:~/hil-serl/examples/experiments/basic_task$ bash run./run_actor.sh
(hilserl4) rwr_team5@SRL-LPT-001:~/hil-serl/examples/experiments/basic_task$ bash ./run_learner.sh