deepspeed \
run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-2.7B \
    --train_file "/gscratch/argon/artidoro/data/nela-covid-2020-train.json" \
    --validation_file "/gscratch/argon/artidoro/data/nela-covid-2020-valid.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "/gscratch/argon/artidoro/transformers/examples/pytorch/language-modeling/gptneo-covid" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --logging_steps 10000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --block_size 32 \
    --bf16 \
    --overwrite_output_dir \
    --deepspeed deepspeed_config.json \
