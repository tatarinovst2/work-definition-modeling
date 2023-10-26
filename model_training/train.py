"""A script for training a T5 model"""
from pathlib import Path

import numpy as np
import torch  # pylint: disable=import-error
from transformers import (  # pylint: disable=import-error
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    PreTrainedTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    TrainerCallback, TrainerControl, TrainerState, TrainingArguments)

from constants import ROOT_DIR
from model_training.dataset_processing import prepare_dataset
from model_training.metrics import get_bleu_score, get_rouge_score
from model_training.utils import plot_metric, plot_training_and_test_loss


def preprocess_function(examples: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """
    Preprocesses the examples for the model using the tokenizer.
    :param examples: dict - the readable examples to preprocess.
    :param tokenizer: PreTrainedTokenizer - a tokenizer that was used to train the model.
    :return: dict - the tokenized examples.
    """
    max_input_length = 1024
    max_target_length = 128

    inputs = list(examples["input_text"])
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=examples["target_text"],
                       max_length=max_target_length,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred: tuple, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
    """
    Computes the metrics for the evaluation.
    :param eval_pred: tuple - the tuple of predictions and labels.
    :param tokenizer: PreTrainedTokenizer - a tokenizer that was used to encode the dataset.
    :return: dict - the metrics.
    """
    predictions, labels = eval_pred

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    results = {
        "blue": get_bleu_score(decoded_predictions, decoded_labels),
        # "bert-f1": get_bert_score(decoded_predictions, decoded_labels)["f1"]
    }
    results.update(get_rouge_score(decoded_predictions, decoded_labels))

    return results


class LossLoggingCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """A callback that draws plots for metrics"""
    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        graphs_dir = checkpoint_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        plot_training_and_test_loss(state.log_history, graphs_dir / "loss-plot-epoch.png",
                                    plot_epochs=True)
        plot_training_and_test_loss(state.log_history, graphs_dir / "loss-plot-step.png",
                                    plot_epochs=False)

        for metric in ["rouge1", "rouge2", "rougeL", "blue"]:
            plot_metric(metric, state.log_history,
                        graphs_dir / f"{metric}-plot-epoch.png", plot_epochs=True)
            plot_metric(metric, state.log_history,
                        graphs_dir / f"{metric}-plot-step.png", plot_epochs=False)


if __name__ == "__main__":
    # MODEL_CHECKPOINT = "ai-forever/FRED-T5-large"
    MODEL_CHECKPOINT = "cointegrated/ruT5-small"

    loaded_tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print(f"Using {type(loaded_tokenizer)}")

    dataset = prepare_dataset(ROOT_DIR / "wiktionary_parser" / "data" / "definitions.jsonl")

    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, loaded_tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")

    BATCH_SIZE = 4
    model_name = MODEL_CHECKPOINT.rsplit('/', maxsplit=1)[-1]
    arguments = Seq2SeqTrainingArguments(
        ROOT_DIR / "models" / f"{model_name}-definition-modeling",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        optim="adafactor",
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        logging_strategy="epoch"
    )

    data_collator = DataCollatorForSeq2Seq(loaded_tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        arguments,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=loaded_tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, loaded_tokenizer),
        callbacks=[LossLoggingCallback()]
    )

    trainer.train()

    training_losses = trainer.state.log_history
    for log in training_losses:
        print(log)
