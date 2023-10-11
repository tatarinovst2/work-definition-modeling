import csv
import json
import sys
import time


def convert_to_csv(json_filepath: str, output_filepath: str):
    with open(json_filepath, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    csv_rows = []

    for word, word_data in json_data.items():
        definitions = word_data.get("definitions", [])
        for definition in definitions:
            value = definition.get("value", "")
            example = definition.get("example", "")
            csv_rows.append([word, example, value])

    with open(output_filepath, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["word", "context", "definition"])
        csv_writer.writerows(csv_rows)


def print_progress_bar(current: int, total: int, start_time, length: int = 50):
    current_time = time.time()
    progress = current / total
    arrow = '=' * int(round(progress * length))
    spaces = ' ' * (length - len(arrow))
    percentage = int(progress * 100)

    if current > 0:
        time_elapsed = current_time - start_time
        estimated_total_time = time_elapsed / current * total
        time_remaining = estimated_total_time - time_elapsed

        # Convert time to minutes and hours if it's too big for seconds
        if time_elapsed > 60:
            time_elapsed_minutes = int(time_elapsed / 60)
            time_elapsed_seconds = int(time_elapsed % 60)
            time_elapsed_str = f'{time_elapsed_minutes}m {time_elapsed_seconds}s'
        else:
            time_elapsed_str = f'{time_elapsed:.2f}s'

        if estimated_total_time > 60:
            estimated_total_time_minutes = int(estimated_total_time / 60)
            estimated_total_time_seconds = int(estimated_total_time % 60)
            estimated_total_time_str = f'{estimated_total_time_minutes}m {estimated_total_time_seconds}s'
        else:
            estimated_total_time_str = f'{estimated_total_time:.2f}s'

        if time_remaining > 60:
            time_remaining_minutes = int(time_remaining / 60)
            time_remaining_seconds = int(time_remaining % 60)
            time_remaining_str = f'{time_remaining_minutes}m {time_remaining_seconds}s'
        else:
            time_remaining_str = f'{time_remaining:.2f}s'

    else:
        time_elapsed_str = '0s'
        estimated_total_time_str = '0s'
        time_remaining_str = '0s'

    sys.stdout.write(f'\r[{arrow + spaces}] {percentage}% ({current}/{total}) - '
                     f'Elapsed: {time_elapsed_str} - Estimated: {estimated_total_time_str} - Remaining: {time_remaining_str}')

    if current >= total:
        sys.stdout.write(f'\n')
    sys.stdout.flush()


if __name__ == "__main__":
    json_filepath = "tmp/definitions.json"
    output_filepath = "tmp/definitions.csv"

    convert_to_csv(json_filepath, output_filepath)
