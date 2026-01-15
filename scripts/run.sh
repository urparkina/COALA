#!/bin/bash
set -e
set -o pipefail 

USE_FP16=${USE_FP16:-False}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1024}

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
    --model_max_length $CONTEXT_LENGTH \
    --config_path $CONFIG_PATH \


# Validate
python3 -u valid.py \
    --path $MODEL_DIR \
    --name $BASE_MODEL \
    --output_dir "$TMP_DIR" \
    --per_device_eval_batch_size 16 \
    --do_train False \
    --do_eval True \
    --fp16 $USE_FP16 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --start_token 349 \
    --model_max_length $CONTEXT_LENGTH \
    --config_path $CONFIG_PATH \
    --name_datasets piqa hellaswag boolq winogrande arc_easy arc_challenge openbookqa
