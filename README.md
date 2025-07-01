<h1>Interpretable approach to detecting semantic changes based on generated definitions</h1>

This is the repository for the paper "Interpretable approach to detecting semantic changes based on generated
definitions" and thesis "Assessment of the applicability of the method of
detecting semantic changes of words by a neural network language model based on generated
definitions" by [Maxim Tatarinov](https://github.com/tatarinovst2)

## PDF files

- [Paper](paper/main.pdf)

- [Poster](poster/main.pdf)

- [Thesis](thesis/main.pdf)

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

## BibTeX

```bibtex
@inproceedings{tatarinov2025interpretable,
  title={Interpretable approach to detecting semantic changes based on generated definitions},
  author={Tatarinov, Maksim and Demidovsky, Aleksandr},
  booktitle={Computational Linguistics and Intellectual Technologies: Proceedings of the International Conference “Dialogue 2025”},
  year={2025}
}
```

```bibtex
@inproceedings{tatarinov2025interpretable-ru,
  title={Интерпретируемый подход к детектированию семантических изменений слов на основе генерируемых определений},
  author={Татаринов, Максим and Демидовский, Александр},
  booktitle={Компьютерная лингвистика и интеллектуальные технологии: по материалам международной конференции «Диалог 2025»},
  year={2025}
}
```
