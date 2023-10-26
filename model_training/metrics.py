"""A script that provides metrics, such as ROUGE or BLEU"""
from evaluate import load
from pymystem3 import Mystem


def get_bleu_score(predictions: list[str], labels: list[str]) -> float:
    """
    Computes the BLEU score for a list of predictions and labels.
    :param predictions: list[str] - the list of predictions.
    :param labels: list[str] - the list of labels.
    :return: float - the BLEU score.
    """
    mystem = Mystem()

    decoded_labels = ["".join(mystem.lemmatize(label)).strip() for label in labels]
    decoded_predictions = ["".join(mystem.lemmatize(prediction)).strip()
                           for prediction in predictions]

    blue_metric = load("sacrebleu")

    blue_results = blue_metric.compute(predictions=decoded_predictions,
                                       references=decoded_labels)

    return blue_results["score"]


def get_rouge_score(predictions: list[str], labels: list[str]) -> dict[str, float]:
    """
    Computes the ROUGE score for a list of predictions and labels.
    :param predictions: list[str] - the list of predictions.
    :param labels: list[str] - the list of labels.
    :return: dict - the scores for ROUGE-1, ROUGE-2 and ROUGE-L.
    """
    mystem = Mystem()

    decoded_labels = ["".join(mystem.lemmatize(label)).strip() for label in labels]
    decoded_predictions = ["".join(mystem.lemmatize(prediction)).strip()
                           for prediction in predictions]

    rouge_metric = load("rouge")

    rouge_results = rouge_metric.compute(predictions=decoded_predictions,
                                         references=decoded_labels,
                                         use_stemmer=False,
                                         tokenizer=lambda x: x.split())

    return {"rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"]}


def get_bert_score(predictions: list[str], labels: list[str]) -> dict[str, float]:
    """
    Computes the BERT score for a list of predictions and labels.
    :param predictions: list[str] - the list of predictions.
    :param labels: list[str] - the list of labels.
    :return: dict - the scores for precision, recall and F1.
    """
    mystem = Mystem()

    decoded_labels = ["".join(mystem.lemmatize(label)).strip() for label in labels]
    decoded_predictions = ["".join(mystem.lemmatize(prediction)).strip()
                           for prediction in predictions]

    bert_metric = load("bertscore")

    bert_results = bert_metric.compute(predictions=decoded_predictions,
                                       references=decoded_labels,
                                       lang="ru")

    return {"precision": bert_results["precision"],
            "recall": bert_results["recall"],
            "f1": bert_results["f1"]}
