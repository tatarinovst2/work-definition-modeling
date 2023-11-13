# Inference

This directory contains the code for inference.

## How to run

If you've already trained the model, you can skip to 

1. Open the Terminal or Command Prompt in the root directory of this repository.

    ```bash
    cd path/to/work-definition-modeling
    ```

2. Make sure you've installed the requirements.
You can do it by running

    ```bash
    pip3 install -r requirements.txt
    pip3 install -r requirements_train.txt
    ```

    You also have to install the `torch` library that will be optimal for your machine.
    For example, if you have a CUDA enabled GPU on Windows, you would probably want to run this:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    Check the [official PyTorch website](https://pytorch.org/get-started/locally/)
    for more information.

3. Make sure you have the checkpoint.
It is a folder containing `pytorch_model.bin` and other files.

If not, follow the instructions in [model_training](../model_training/README.md).

## Two ways to run inference

There are two ways to run inference:

### On a dataset

You must have a `.jsonl` file with a `prompt` field in each line.

Supply the path to the file as an argument to `inference.py`:

```bash
python3 -m inference.inference path/to/checkpoint --input_filepath path/to/dataset.jsonl --output_filepath path/to/output.jsonl
```

In this case, the output will be a `.jsonl` file with the same format as the input file,
but with the `generated_text` field added to each line.

### On a single prompt

Run `inference.py` without the `--input_filepath` argument:

```bash
python3 -m inference.inference path/to/checkpoint
```

Then, you will have to enter a prompt.
The output will be printed to the console.

You can exit the program by entering an empty line or pressing Ctrl+C.
