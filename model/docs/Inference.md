# Inference

This directory contains the code for inference.

## How to run

1. Prerequisites

    Make sure you've installed the requirements and opened the Terminal or Command prompt
    in the root directory of this repository.

    If not, follow the first two instructions in [Model training](Model%20training.md).

2. Make sure you have the checkpoint.

    It is a folder containing `pytorch_model.bin` and other files.

    If not, follow the other instructions in [Model training](Model%20training.md).

## Two ways to run inference

There are two ways to run inference:

### On a dataset

You must have a `.jsonl` file with a input field in each line.
By default, it is `input_text` field.

Such a dataset is created by the `model/train.py` script. It corresponds
to the test split of the dataset.

Supply the path to the file as an argument to `inference.py`:

```bash
python3 model/inference.py path/to/checkpoint
--input-file path/to/dataset.jsonl --output-file path/to/output.jsonl
```

In this case, the output will be a `.jsonl` file with the same format as the input file,
but with the `generated_text` field added to each line.

### On a single prompt

Run `inference.py` without the `--input_file` argument:

```bash
python3 model/inference.py path/to/checkpoint
```

Then, you will have to enter a prompt.
The output will be printed to the console.

You can exit the program by entering an empty line or pressing Ctrl+C.
