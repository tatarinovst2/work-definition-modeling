#!/bin/bash

lora_file_path=""
batch_size=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size)
            batch_size="--batch_size $2"
            shift
            shift
            ;;
        *)
            if [ -z "$base_model_name_or_path" ]; then
                base_model_name_or_path="$1"
            elif [ -z "$lora_file_path" ]; then
                lora_file_path="-l $1"
            fi
            shift
            ;;
    esac
done

model_name=$(basename "$base_model_name_or_path")

python3 model/inference.py "$base_model_name_or_path" $lora_file_path $batch_size \
    --input-file rushifteval/tmp/for_inference/rusemshift/rusemshift_all_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rusemshift_all.jsonl" \

python3 model/inference.py "$base_model_name_or_path" $lora_file_path $batch_size \
    --input-file rushifteval/tmp/for_inference/rushifteval/rushifteval1_test_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rushifteval1_test.jsonl"

python3 model/inference.py "$base_model_name_or_path" $lora_file_path $batch_size \
    --input-file rushifteval/tmp/for_inference/rushifteval/rushifteval2_test_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rushifteval2_test.jsonl"

python3 model/inference.py "$base_model_name_or_path" $lora_file_path $batch_size \
    --input-file rushifteval/tmp/for_inference/rushifteval/rushifteval3_test_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rushifteval3_test.jsonl"
