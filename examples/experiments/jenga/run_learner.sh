export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=jenga \
    --checkpoint_path=second_run \
    --demo_path=/home/rwr_team5/hil-serl/examples/experiments/jenga/demo_data/jenga_20_demos_2025-12-12_12-12-59.pkl \
    --learner \
    --debug \