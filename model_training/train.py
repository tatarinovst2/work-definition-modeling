"""A script for training a T5 model"""
from pathlib import Path

import numpy as np
import torch  # pylint: disable=import-error
from datasets import DatasetDict
from transformers import (  # pylint: disable=import-error
    AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding,
    DataCollatorForSeq2Seq, PreTrainedTokenizer, Seq2SeqTrainer,
    Seq2SeqTrainingArguments, TrainerCallback, TrainerControl, TrainerState,
    TrainingArguments)

from constants import ROOT_DIR
from model_training.dataset_processing import prepare_dataset
from model_training.metrics import get_bleu_score, get_rouge_score
from model_training.utils import (load_train_config, plot_metric,
                                  plot_training_and_test_loss)


def preprocess_function(examples: dict, tokenizer: PreTrainedTokenizer) -> BatchEncoding:
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
    """
    A callback that draws plots for metrics
    """
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
    train_config = load_train_config(ROOT_DIR / "model_training" / "train_config.json")

    loaded_tokenizer = AutoTokenizer.from_pretrained(train_config["model_checkpoint"])
    print(f"Using {type(loaded_tokenizer)}")

    dataset_dict = prepare_dataset(ROOT_DIR / train_config["dataset_path"])

    if train_config.get("debug", False):
        new_dataset_dict = {}
        for split in dataset_dict:
            new_dataset_dict[split] = dataset_dict[split].select(range(100))
        dataset_dict = DatasetDict(new_dataset_dict)

    tokenized_datasets = dataset_dict.map(
        lambda examples: preprocess_function(examples, loaded_tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(train_config["model_checkpoint"])

    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")

    model_name = train_config["model_checkpoint"].rsplit('/', maxsplit=1)[-1]
    arguments = Seq2SeqTrainingArguments(
        ROOT_DIR / "models" / f"{model_name}-definition-modeling",
        learning_rate=train_config["learning_rate"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        gradient_checkpointing=train_config.get("gradient_checkpointing", False),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        weight_decay=train_config.get("weight_decay", 0.01),
        optim=train_config.get("optimizer", "adamw_torch"),
        save_total_limit=train_config.get("save_total_limit", 3),
        evaluation_strategy=train_config.get("evaluation_strategy", "epoch"),
        save_strategy=train_config.get("save_strategy", "epoch"),
        logging_strategy=train_config.get("logging_strategy", "epoch"),
        save_steps=train_config.get("save_steps", 500),
        logging_steps=train_config.get("logging_steps", 500),
        eval_steps=train_config.get("eval_steps", 500),
        num_train_epochs=train_config.get("num_train_epochs", 3),
        max_steps=train_config.get("max_steps", -1),
        predict_with_generate=train_config.get("predict_with_generate", False),
        fp16=train_config.get("fp16", False),
        push_to_hub=train_config.get("push_to_hub", False)
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
