#!/bin/bash

predictions_folder="t5_1.7b_3_epoch_rushemeval_preds"
vectors_folder="t5_1.7b_3_epoch_xlm-r_rushemeval_vectors"
model_name="aditeyabaral/sentencetransformer-xlm-roberta-base"

python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rusemshift_1.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rusemshift_1.jsonl" --model_name "${model_name}"
python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rusemshift_2.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rusemshift_2.jsonl" --model_name "${model_name}"
