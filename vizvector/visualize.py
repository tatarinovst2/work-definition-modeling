"""Module to visualize shifts in meaning in the inferred and vectorized dataset."""
import argparse
import json
import textwrap
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic.dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

BASE_FONT_SIZE = 10
DECREMENT_STEP = 0.1


@dataclass
class WordUsage:
    """
    A dataclass representing a row in the JSONL dataset.

    :param word: The target word.
    :param sent: The sentence.
    :param definition: The generated definition for the word usage.
    :param vect: Embedding for the usage.
    """

    word: str
    sent: str
    date: Optional[str] = None
    definition: Optional[str] = None
    vect: Optional[list[float]] = None


def load_data(input_paths: list[str], target_word: str) -> list[WordUsage]:
    """
    Load word usages from the JSONL files for a given target word.

    :param input_paths: The paths to the input JSONL files.
    :param target_word: The target word to filter and cluster.
    :return: The loaded usages for the target word.
    """
    all_data = []

    for input_path in input_paths:
        data = []
        with open(input_path, 'r', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                if item['word'] == target_word:
                    definition = item['generated_text']
                    if not definition.endswith("."):
                        definition += "."

                    word_usage = WordUsage(
                        word=item['word'],
                        sent=item['sentence'],
                        date=item['date'] if 'date' in item else input_path,
                        definition=definition,
                        vect=item.get('vector', [])
                    )
                    data.append(word_usage)
        all_data.extend(data)

    return all_data


def count_vectors_per_cluster(unique_dates: list[str], clusters: np.ndarray,
                              date_mapping: list[int], relative: bool) -> dict:
    """
    Count the number of vectors assigned to each cluster including noise.

    :param unique_dates: A list with time periods.
    :param clusters: A 1D numpy array of cluster labels corresponding to each row in `data`.
    :param date_mapping: Mapping of usages to unique_dates.
    :param relative: Show the values in percentages relative to all usages in the epoch.
    :return: A dictionary with cluster labels as keys and counts as values.
    """
    date_clusters: dict[str, list[int]] = {date: [] for date in unique_dates}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id != -1:
            date_clusters[unique_dates[date_mapping[idx]]].append(cluster_id)

    date_cluster_counts = {}
    for date, cluster_list in date_clusters.items():
        unique, counts = np.unique(cluster_list, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        if relative:
            total_counts = sum(counts)
            for cluster_id in counts_dict:
                counts_dict[cluster_id] = counts_dict[cluster_id] / total_counts * 100

        date_cluster_counts[date] = counts_dict

    return date_cluster_counts


def count_outliers(clusters: np.ndarray) -> tuple[int, float]:
    """
    Count the number of outliers in the clusters data.

    :param clusters: A 1D numpy array of cluster labels corresponding to each row in `data`.
    :return: A tuple containing the absolute and relative count.
    """
    count = 0

    for cluster in np.unique(clusters):
        if cluster == -1:
            cluster_points_idx = np.where(clusters == cluster)[0]
            count = len(cluster_points_idx)

    return count, count / len(clusters)


def find_representative_index(data: np.ndarray, clusters: np.ndarray,
                              method: str = 'centroid') -> dict:
    """
    Find the representative index for each cluster based on the specified method.

    :param data: A 2D numpy array where each row represents a vector.
    :param clusters: A 1D numpy array of cluster labels corresponding to each row in `data`.
    :param method: The method to use for finding the representative index ('centroid' or 'medoid').
    :return: A dictionary with cluster labels as keys and the index of the vector as values.
    :raises ValueError: If the method name is incorrect.
    """
    if method not in ['centroid', 'medoid']:
        raise ValueError("Method must be 'centroid' or 'medoid'")

    representative_idx = {}
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        cluster_points_idx = np.where(clusters == cluster)[0]
        cluster_points = data[cluster_points_idx]

        if method == 'centroid':
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            min_idx = np.argmin(distances)
            representative_idx[cluster] = cluster_points_idx[min_idx]
        elif method == 'medoid':
            pairwise_dist = pairwise_distances(cluster_points, metric="cosine")
            medoid_idx = np.argmin(np.sum(pairwise_dist, axis=0))
            representative_idx[cluster] = cluster_points_idx[medoid_idx]

    return representative_idx


def cluster_data_and_get_definitions(all_data: list[WordUsage],
                                     eps: float = 0.5, relative: bool = True,
                                     min_samples: int = 5,
                                     do_normalize: bool = False) -> tuple[dict, dict]:
    """
    Cluster the data and get the definitions for the clusters.

    :param all_data: The data to cluster.
    :param eps: Maximum distance for considering neighborhood.
    :param relative: Show the values in percentages relative to all usages in the epoch.
    :param min_samples: Minimum samples for a core point.
    :param do_normalize: Normalize the vectors.
    :return: The date-cluster counts and the definitions.
    """
    unique_dates = sorted(list(set(usage.date for usage in all_data
                                   if usage.vect is not None and usage.date is not None)))
    date_mapping = [unique_dates.index(usage.date) for usage in all_data
                    if usage.vect is not None and usage.date is not None]
    aggregated_vectors = np.array([usage.vect for usage in all_data if usage.vect is not None])

    if do_normalize:
        aggregated_vectors = normalize(aggregated_vectors, axis=1, norm='l2')

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(aggregated_vectors)

    representative_indices = find_representative_index(aggregated_vectors, clusters,
                                                       method='centroid')

    definitions = {cluster: all_data[idx].definition
                   for cluster, idx in representative_indices.items()}

    date_cluster_counts = count_vectors_per_cluster(unique_dates, clusters, date_mapping,
                                                    relative=relative)

    outliers_count, outliers_percentage = count_outliers(clusters)

    print(f"Outliers count: {outliers_count}, outliers percentage: {outliers_percentage}")

    return date_cluster_counts, definitions


def get_definitions_subplot_text(definitions: dict) -> str:
    """
    Get the text for the subplot with definitions.

    :param definitions: The definitions.
    :return: The text for the subplot.
    """
    definitions_text = ""
    for idx, (_, meaning) in enumerate(definitions.items()):
        wrapped_text = textwrap.fill(f"{idx + 1}: {meaning}", width=70)
        definitions_text += wrapped_text + "\n"
    return definitions_text


def visualize_changes_with_legend(date_cluster_counts: dict,  # pylint: disable=too-many-locals
                                  definitions: dict,
                                  eps: float,
                                  min_samples: int,
                                  relative: bool = True) -> None:
    """
    Visualize semantic changes.

    :param date_cluster_counts: The counts of clusters for each date.
    :param definitions: The definitions for the clusters.
    :param eps: Maximum distance for considering neighborhood.
    :param min_samples: Minimum samples for a core point.
    :param relative: Show the values in percentages relative to all usages in the epoch.
    """
    dates = list(date_cluster_counts.keys())
    max_meaning_id = max(max(counts.keys()) for counts in date_cluster_counts.values()) + 1

    total_width = 0.8
    bar_width = total_width / len(dates)

    index = np.arange(max_meaning_id) * (total_width + 0.2)

    colors = plt.cm.Blues(  # type: ignore  # pylint: disable=no-member
        np.linspace(0.2, 1, len(dates)))

    fig, axs = plt.subplots(2, 1, figsize=(7, 7),
                            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.6})
    ax = axs[0]
    for i, (date, counts) in enumerate(date_cluster_counts.items()):
        date_counts = [counts.get(meaning_id, 0) for meaning_id in range(max_meaning_id)]
        ax.bar(index + i * bar_width, date_counts, bar_width, label=f'{date}', color=colors[i])

    ax.set_xlabel('Значения')
    ax.set_ylabel('Процент от всех использований' if relative else 'Частота')
    ax.set_xticks(index + total_width / 2 - bar_width / 2)
    ax.set_xticklabels([f'{i + 1}' for i in range(max_meaning_id)])

    leg = ax.legend(title="Периоды", loc='upper left')
    leg.get_frame().set_alpha(0.5)

    definitions_text = get_definitions_subplot_text(definitions)

    adjusted_font_size = max(BASE_FONT_SIZE * (1 - max(len(definitions) - 8, 0) * DECREMENT_STEP),
                             8)

    axs[1].text(0.01, 0.5, definitions_text, verticalalignment='center',
                horizontalalignment='left', fontsize=adjusted_font_size, wrap=True)
    axs[1].set_axis_off()

    param_description = f"Параметры: eps={eps}, min_samples={min_samples}"
    fig.text(0.02, 0.02, param_description, fontsize=9, verticalalignment='bottom',
             horizontalalignment='left')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Visualize semantic changes."""
    parser = argparse.ArgumentParser(
        description="Visualize semantic change across multiple epochs.")
    parser.add_argument("input_paths", type=str, nargs='+',
                        help="Paths to the input JSONL files.")
    parser.add_argument("word", type=str,
                        help="Target word to filter and cluster.")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="The maximum distance between two samples for one "
                             "to be considered as in the neighborhood of the other.")
    parser.add_argument("--relative", type=bool, default=True,
                        help="Show the values in percentages relative "
                             "to all usages in the epoch.")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="The number of samples in a neighborhood for a "
                             "point to be considered as a core point")
    parser.add_argument("--do-normalize", type=bool, default=True,
                        help="Normalize the vectors.")
    args = parser.parse_args()

    all_data = load_data(args.input_paths, args.word)
    epoch_cluster_counts, definitions = cluster_data_and_get_definitions(all_data, eps=args.eps,
                                                                         relative=args.relative,
                                                                         min_samples=
                                                                         args.min_samples,
                                                                         do_normalize=
                                                                         args.do_normalize)
    visualize_changes_with_legend(epoch_cluster_counts, definitions,
                                  relative=args.relative, eps=args.eps,
                                  min_samples=args.min_samples)


if __name__ == "__main__":
    main()
