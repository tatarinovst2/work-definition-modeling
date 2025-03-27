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
  * [Inference](model/docs/Inference.md) - generate definitions for the dataset.
  * [Evaluation](model/docs/Evaluation.md) - evaluate the model.

## Steps to reproduce

To reproduce the results, follow the steps below. Steps can be skipped if you download the corresponding data.

1. The instructions for parsing MAS dataset are in the
[README](mas_parser/README.md) of the `mas_parser` module.
2. The instructions for training the model, initial testing and inference are in the
[README](model/docs/Model%20training.md) of the `model` module.
3. The instructions for evaluating the model with `rushifteval` are in the
[README](rushifteval/README.md) of the `rushifteval` module.
4. The instructions for working with RNC (visualizing the words' meanings) are in the
[README](ruscorpora/README.md) of the `ruscorpora` module.
