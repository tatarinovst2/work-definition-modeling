# Rushifteval module

This is a module for checking the quality of using the fine-tuned T5 model for
detecting semantic change.

## How to run

### 1. Download `data`

Download the `data` archive, unpack it and put the contents under `rushifteval` directory:
[Download](https://github.com/tatarinovst2/work-definition-modeling/issues/21)

You'll have a `rushifteval/data` folder with `gold`, `rushifteval`, and `rusemshift` folders
inside.

The meaning of the folders:

- `gold` contains the gold correlation coefficients against which we will check the quality later.
- `rushifteval` contains the dataset for which we will calculate the coefficient.
- `rusemshift` is a training dataset, e.g. we will use it train models converting distances
between vectors to the dataset format.

### 2. Process annotations and get definitions

> NOTE: Inference takes time.
> You can skip this step by downloading the archive and
> unpacking it under `rushifteval/data` directory:
> [Download](https://github.com/tatarinovst2/work-definition-modeling/issues/23)

#### 2.1 Process annotations

Since we need to get definitions for each sentence pair, we need to run inference on the dataset.
But before that we must convert the `.tsv` files to `.jsonl` files that `inference.py` can use.

`process_raw_annotations.py` can be run like this:

```bash
python rushifteval/process_raw_annotations.py path_to_the_raw_dataset path_to_the_new_dataset
```

The `path_to_the_raw_dataset` can be either a `.tsv` file or a directory containing such files.

Run `process_raw_annotations.sh` to run commands needed for `rushifteval` and `rusemeval`:

```bash
bash rushifteval/bash/process_raw_annotations.sh
```

#### 2.2 Get definitions

Now we have the `.jsonl` files that `inference.py` can use.

Run the `get_definitions.sh` script to get the definitions.

You have to supply the path to the model like this:

```bash
bash rushifteval/bash/get_definitions.sh path_to_the_model
```

If you have a LoRa adapter, supply the base model name and the path to the adapter, e.g.:

```bash
bash rushifteval/bash/get_definitions.sh ai-forever/FRED-T5-large \
models/checkpoint-fred-t5-large-lora-30000
```

The optional argument is `batch_size`. Use it like this to make inference faster.

```bash
bash rushifteval/bash/get_definitions.sh base_model_name path_to_the_adapter --batch_size 16
```

A folder will appear under `rushifteval/data/preds` with the definitions.

#### PS: You can download existing LoRa adapters

> 23 December FRED-T5-large 2.87 epochs
> [Download](https://github.com/tatarinovst2/work-definition-modeling/issues/15)

> 18 January FRED-T5-1.7B 3 epochs
> [Download](https://github.com/tatarinovst2/work-definition-modeling/issues/22)

### 3. Vectorize the definitions

Establish the name of the folder inside `rushifteval/data/preds` that contains the definitions
and which you want to vectorize.

Pass it together with the sentence-transformer model to the `rushifteval/bash/vectorize.sh` file.

For example:

```bash
bash rushifteval/bash/vectorize.sh FRED-T5-1.7B_preds cointegrated/rubert-tiny2
```

### 4. Calculate the scores

Run `rushifteval/bash/get_scores.sh`:

For example:

```bash
bash rushifteval/bash/get_scores.sh FRED-T5-1.7B_preds_rubert-tiny2 cosine --normalize
```

where the first argument is the name of the folder with vectors, the second
argument is the metric and the third optional argument is whether to
normalize the vectors.

The script will report the results in the following format:

```text
Mean correlation: 0.6411617197963112
Correlations for all epoch pairs:
pre-Soviet:Soviet: 0.6128661193246704
Soviet:post-Soviet: 0.670315740275251
pre-Soviet:post-Soviet: 0.640303299789012
```
