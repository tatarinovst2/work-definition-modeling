# Inference

This directory contains the code for inference.

## How to run

1. Prerequisites

    Make sure you've installed the requirements and opened the Terminal or Command prompt
    in the root directory of this repository.

    If not, follow the first two instructions in [Model training](Model%20training.md).

2. Make sure you have the checkpoint.

    It is a folder containing either a `pytorch_model.bin` file in case you fine-tuned the whole
    model or `adapter_model.safetensors` if you used LoRa.

    If not, follow the other instructions in [Model training](Model%20training.md).

### You can download existing LoRa adapters:

> 23 December FRED-T5-large 2.87 epochs
> [Download](https://github.com/tatarinovst2/work-definition-modeling/issues/15)

Unpack it and put under `models` directory.

## Two ways to run inference

There are two ways to run inference:

### On a split dataset

You must have a `.jsonl` file with an input field in each line.
By default, it is `input_text` field.
Such files (`test.jsonl`, `train.jsonl`, `val.jsonl`) are created after running `split_dataset.py`.

> NOTE: You can also download existing splits:

> 23 December splits
> [Download](https://github.com/tatarinovst2/work-definition-modeling/issues/18)

> Unpack and put it under `model/data/splits`. Learn more in [Model training](Model%20training.md).

Supply the path to the file as an argument to `inference.py`:

```bash
python model/inference.py path/to/checkpoint
--input-file path/to/dataset.jsonl --output-file path/to/output.jsonl
```

Optionally use `--lora-checkpoint` or `-l` argument if you want to load a LoRa checkpoint.
You can load the base model using its hugging-face name or by providing the path to the checkpoint.

For example:

```bash
python model/inference.py ai-forever/FRED-T5-large -l models/checkpoint-fred-t5-large-lora-30000
--input-file model/data/cleaned_splits/test.jsonl --output-file model/data/output/output_test.jsonl
```

In this case, the output will be a `.jsonl` file with the same format as the input file,
but with the `generated_text` field added to each line.

> NOTE: Inference with a batch size of more than 1 seems to be bugged for `mps` pipeline,
> but should work with `cuda`.

### On a single prompt

Run `inference.py` without the `--input_file` argument:

```bash
python model/inference.py path/to/checkpoint
```

Then, you will have to enter a prompt.
The output will be printed to the console.

You can exit the program by entering an empty line or pressing Ctrl+C.
