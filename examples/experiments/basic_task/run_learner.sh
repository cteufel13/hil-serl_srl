export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=basic_task \
    --checkpoint_path=first_run_maxim \
    --demo_path=/home/rwr_team5/hil-serl/examples/experiments/basic_task/demo_data/basic_task_20_demos_2025-12-11_11-04-50.pkl \
    --learner \
    --debug \