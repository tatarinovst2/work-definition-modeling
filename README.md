<h1>Thesis "Assessment of the applicability of the method of detecting semantic changes of words
by a neural network language model based on generated definitions"</h1>

This is the repository for the thesis "Assessment of the applicability of the method of
detecting semantic changes of words by a neural network language model based on generated
definitions" by [Maxim Tatarinov](https://github.com/tatarinovst2)

## Components

The repository consists of the following components:

* [wiktionary_parser](wiktionary_parser/README.md) - a parser for Wiktionary dump files.
* [model](model) - a module for training, evaluation and inference of the FRED-T5-1.7B model.
  * [Model training](model/docs/Model%20training.md) - split the dataset and train the model.
  * [Inference](model/docs/Inference.md) - run inference server or on a dataset.
  * [Evaluation](model/docs/Evaluation.md) - evaluate the model on the inferred dataset.

## Prepared assets

You can download some prepared assets from this
[link](https://drive.google.com/drive/folders/1D715SIIWZMgQIwCABcPpsNbrF7mXppsz?usp=sharing).

`definitions.jsonl` - raw dataset not yet split into test, train and val.
Put under `wiktionary_parser/data`.

`splits` - the dataset split into parts. Put under `model/data`.
