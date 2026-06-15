python3 run.py \
    --task text \
    --task_start_index 0 \
    --task_end_index 200 \
    --method_generate sample \
    --method_evaluate vote \
    --method_select greedy \
    --n_generate_sample 5 \
    --n_evaluate_sample 5 \
    --n_select_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    ${@}
