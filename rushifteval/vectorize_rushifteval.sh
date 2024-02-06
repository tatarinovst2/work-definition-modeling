#!/bin/bash

predictions_folder="t5_1.7b_3_epoch_rushifteval_preds"
vectors_folder="t5_1.7b_3_epoch_xlm-r_rushifteval_vectors"
model_name="aditeyabaral/sentencetransformer-xlm-roberta-base"

python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rushifteval1_dev.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rushifteval1_dev.jsonl" --model_name "${model_name}"
python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rushifteval1_test.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rushifteval1_test.jsonl" --model_name "${model_name}"
python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rushifteval2_dev.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rushifteval2_dev.jsonl" --model_name "${model_name}"
python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rushifteval2_test.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rushifteval2_test.jsonl" --model_name "${model_name}"
python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rushifteval3_dev.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rushifteval3_dev.jsonl" --model_name "${model_name}"
python rushifteval/vectorize.py "rushifteval/data/predictions/${predictions_folder}/preds_rushifteval3_test.jsonl" "rushifteval/data/vectors/${vectors_folder}/vectors_rushifteval3_test.jsonl" --model_name "${model_name}"
