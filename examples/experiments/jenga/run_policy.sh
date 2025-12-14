export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=jenga \
    --checkpoint_path=second_run \
    --eval_checkpoint_step=170000 \
    --eval_n_trajs=1000 \
    --actor \