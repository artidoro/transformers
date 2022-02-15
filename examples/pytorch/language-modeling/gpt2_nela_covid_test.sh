python run_clm.py \
    --model_name_or_path gpt2 \
    --validation_file "/gscratch/argon/artidoro/data/nela-covid-2020-test.json" \
    --do_eval \
    --logging_first_step \
    --output_dir "/gscratch/argon/artidoro/transformers/examples/pytorch/language-modeling/gpt2-covid-test" \
    --per_device_eval_batch_size 16 \
    --block_size 512 \
