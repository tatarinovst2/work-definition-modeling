import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def load_model(model_checkpoint: str | Path) -> tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """
    Load a model from a checkpoint.

    :param model_checkpoint: The path to the checkpoint.
    :return: The model and the tokenizer.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, eos_token='</s>')

    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")

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


def save_output(output_text: str, output_file: str | Path, initial_json: dict) -> None:
    """
    Appends generated text to the output file.

    :param output_text: The generated text.
    :param output_file: The output file of JSON Lines format.
    :param initial_json: The initial JSON object.
    """
    with open(output_file, "a", encoding="utf-8") as output_file:
        initial_json["generated_text"] = output_text
        output_file.write(json.dumps(initial_json) + "\n")


def main():
    """Entry point."""

    parser = argparse.ArgumentParser(
        description="Process prompts from the command line or a JSON lines file.")

    parser.add_argument("model_checkpoint",
                        type=str,
                        help="The path to the model checkpoint.")
    parser.add_argument("--input_file",
                        type=str,
                        help="The input file of JSON Lines format.")
    parser.add_argument("--output_file",
                        type=str,
                        help="The output file of JSON Lines format.")

    args = parser.parse_args()

    validate_args(args)

    model, tokenizer = load_model(args.model_checkpoint)

    if args.input_file is not None:
        with open(args.input_file, "r", encoding="utf-8") as input_file:
            for line in input_file:
                json_object = json.loads(line)
                prompt = json_object["prompt"]
                output_text = run_inference(model, tokenizer, prompt)
                save_output(output_text, args.output_file, json_object)
    else:
        while True:
            prompt = input("Enter a prompt: ")

            if not prompt:
                print("Exiting...")
                exit(0)

            output_text = run_inference(model, tokenizer, prompt)
            print(output_text, end="\n\n")


if __name__ == "__main__":
    main()

