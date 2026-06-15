python3 run.py \
    --task text \
    --task_start_index 0 \
    --task_end_index 200 \
    --naive_run \
    --prompt_sample standard \
    --n_generate_sample 10 \
    --temperature 1.0 \
    ${@}
