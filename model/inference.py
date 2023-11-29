"""A module for the inference of the model."""
import argparse
import json
import sys
from pathlib import Path

import torch  # pylint: disable=import-error
from src.utils import get_current_torch_device, parse_path
from transformers import AutoTokenizer, T5ForConditionalGeneration  # pylint: disable=import-error


def load_model(model_checkpoint: str | Path) -> tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """
    Load a model from a checkpoint.

    :param model_checkpoint: The path to the checkpoint.
    :return: The model and the tokenizer.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model.to(get_current_torch_device())

    return model, tokenizer


def run_inference(model: T5ForConditionalGeneration, tokenizer: AutoTokenizer,
                  input_text: str, max_length: int = 100) -> str:
    """
    Run inference on a model.

    :param model: The model.
    :param tokenizer: The tokenizer.
    :param input_text: The input text.
    :param max_length: The maximum length of the output.
    :return: The generated text.
    """
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(model.device)

    output_ids = model.generate(input_ids,
                                eos_token_id=tokenizer.eos_token_id,
                                early_stopping=True,
                                max_length=max_length,
                                num_beams=3)

    output_text = tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)

    return output_text


def run_inference_over_dataset(  # pylint: disable=too-many-arguments
        model: T5ForConditionalGeneration,
        tokenizer: AutoTokenizer,
        data: list[dict[str, str]],
        input_field: str,
        output_file_path: str | Path,
        max_length: int = 100) -> None:
    """
    Run inference over a dataset.

    :param model: The model.
    :param tokenizer: The tokenizer.
    :param data: The dataset.
    :param input_field: The name of the field in the dataset that contains the input text.
    :param output_file_path: The path to the output file.
    :param output_field: The name of the field in the dataset that contains the generated text.
    :param max_length: The maximum length of the output.
    """
    entries_inferred = 0
    total_entries = len(data)

    for entry in data:
        prompt = entry[input_field]
        output_text = run_inference(model, tokenizer, prompt, max_length=max_length)
        save_output(output_text, output_file_path, entry)
        entries_inferred += 1
        print(f"\rInferred {entries_inferred} out of {total_entries}.", end="")


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate the arguments.

    Either both input_file and output_file are specified, or none.
    The model_checkpoint must always be specified.
    :param args: The arguments.
    :raises ValueError: If the arguments are invalid.
    """
    if args.input_file and not args.output_file:
        raise ValueError("If input_file is specified, output_file must be specified too.")
    if not args.input_file and args.output_file:
        raise ValueError("If output_file is specified, input_file must be specified too.")
    if not args.model_checkpoint:
        raise ValueError("model_checkpoint must be specified.")


def save_output(output_text: str, output_save_filepath: str | Path, initial_json: dict) -> None:
    """
    Append generated text to the output file.

    :param output_text: The generated text.
    :param output_save_filepath: The output file of JSON Lines format.
    :param initial_json: The initial JSON object.
    """
    output_save_filepath = parse_path(output_save_filepath)

    with open(output_save_filepath, "a", encoding="utf-8") as output_save_file:
        initial_json["generated_text"] = output_text
        output_save_file.write(json.dumps(initial_json, ensure_ascii=False) + "\n")


def load_dataset_for_inference(input_file_path: str | Path, input_field: str,
                               debug: bool = False) -> list[dict[str, str]]:
    """
    Load the dataset for inference.

    :param input_file_path: The path to the input file.
    :param input_field: The name of the field in the dataset that contains the input text.
    :param debug: Whether to load only a small subset of the dataset.
    :return: The dataset.
    :raises ValueError: If the input field is not found in the JSON object.
    """
    data = []

    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            json_object = json.loads(line)
            if input_field not in json_object:
                raise ValueError(f"Field {input_field} not found in JSON object.")
            data.append(json_object)

            if debug and len(data) == 100:
                break

    return data


def validate_output_file(output_file_path: str | Path) -> None:
    """
    Validate the output file. If it exists, ask the user whether to remove it.

    :param output_file_path: The path to the output file.
    """
    output_save_filepath = parse_path(output_file_path)

    if output_save_filepath.exists():
        remove_output = input("Output file already exists. Remove? (y/n) ")
        if remove_output.lower() == "y":
            output_save_filepath.unlink()
        else:
            print("Exiting...")
            sys.exit(0)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Process prompts from the command line or a JSON lines file.")

    parser.add_argument("model_checkpoint",
                        type=str,
                        help="The path to the model checkpoint.")
    parser.add_argument("--input-file", "-i",
                        type=str,
                        help="The input file of JSON Lines format.")
    parser.add_argument("--output-file", "-o",
                        type=str,
                        help="The output file of JSON Lines format.")
    parser.add_argument("--input-field",
                        type=str,
                        default="input_text",
                        help="The name of the field in the dataset that contains the input text.")
    parser.add_argument("--debug", "-d",
                        action="store_true",
                        help="Run in debug mode. (default: False)")

    args = parser.parse_args()

    validate_args(args)

    model, tokenizer = load_model(args.model_checkpoint)

    input_file_path = parse_path(args.input_file)
    output_file_path = parse_path(args.output_file)

    validate_output_file(output_file_path)

    if args.input_file is not None:
        data = load_dataset_for_inference(input_file_path, args.input_field, args.debug)

        run_inference_over_dataset(model, tokenizer, data, args.input_field,
                                   output_file_path)
    else:
        while True:
            prompt = input("Enter a prompt: ")

            if not prompt:
                print("Exiting...")
                sys.exit(0)

            output_text = run_inference(model, tokenizer, prompt)
            print(output_text, end="\n\n")


if __name__ == "__main__":
    main()
