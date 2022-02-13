#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pretrain-any2-gptneo
#SBATCH --partition=ckpt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=artidoro@uw.edu


source activate /gscratch/argon/artidoro/miniconda3/envs/transformers
module load cuda/11.4.1

cat $0
echo "--------------------"

python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-2.7B \
    --train_file "/gscratch/argon/artidoro/data/nela-covid-2020-train.json" \
    --validation_file "/gscratch/argon/artidoro/data/nela-covid-2020-valid.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "/gscratch/argon/artidoro/transformers/examples/pytorch/language-modeling/gptneo-covid" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --save_total_limit 3
