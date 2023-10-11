from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, GPT2Tokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)


def prepare_row(row: dict) -> dict:
    input_text = f"<LM>{row['context']}\n Определение слова {row['word']}: "
    target_text = row['definition']
    return {"input_text": input_text, "target_text": target_text}


def prepare_dataset(filepath: str | Path):
    df = pd.read_csv(filepath)
    prepared_rows = [prepare_row(row) for _, row in df.iterrows()]
    new_df = pd.DataFrame(prepared_rows)

    print(new_df.head())

    created_dataset = Dataset.from_pandas(new_df)
    created_dataset = created_dataset.train_test_split(test_size=0.2)

    return created_dataset


def preprocess_function(examples):
    max_input_length = 1024
    max_target_length = 128

    inputs = [doc for doc in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["target_text"],
                       max_length=max_target_length,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    metric = load("rouge")

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    # model_checkpoint = "ai-forever/FRED-T5-large"
    model_checkpoint = "cointegrated/rut5-small"

    if "FRED" not in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        print("Using AutoTokenizer")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
        print("Using GPT2Tokenizer")

    dataset = prepare_dataset("tmp/definitions.csv")

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")

    batch_size = 4
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-definition-modeling",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        optim="adafactor",
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        logging_steps=100
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    training_losses = trainer.state.log_history
    for log in training_losses:
        print(log)
