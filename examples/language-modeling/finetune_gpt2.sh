python run_clm.py \
    --model_name_or_path gpt2_config_long.json \
    --train_file training_data.txt \
    --validation_file dev_data_trunc.txt \
    --do_train \
    --do_eval \
    --output_dir gpt2-finetune-long \
    --overwrite_output \
    --per_device_train_batch_size 2\
    --n_positions 7500\
    --num_epochs 5\

# --dataset_name wikitext
# --dataset_config_name wikitext-2-raw-v1
    

