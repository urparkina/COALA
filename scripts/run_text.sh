#!/bin/bash
set -e
set -o pipefail 

USE_FP16=${USE_FP16:-False}

# Compress
python3 -u compress.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $MODEL_DIR \
    --per_device_eval_batch_size 2 \
    --do_train False \
    --do_eval True \
    --fp16 $USE_FP16 \
    --report_to "tensorboard" \
    --start_token 349 \
    --model_max_length 2048 \
    --config_path $CONFIG_PATH \


# Validate
python3 -u valid.py \
    --path $MODEL_DIR \
    --name $BASE_MODEL \
    --output_dir "$TMP_DIR" \
    --per_device_eval_batch_size 2 \
    --do_train False \
    --do_eval True \
    --fp16 $USE_FP16 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --start_token 349 \
    --model_max_length 1024 \
    --config_path $CONFIG_PATH \
    --name_datasets mmlu wikitext mathqa
