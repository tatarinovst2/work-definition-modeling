# Rushifteval module

This is a module for checking the quality of using the fine-tuned T5 model for
detecting semantic change.

It uses 

## Process annotations

Since we need to get definitions for each sentence pair, we need to run inference on the dataset.
But before that we must convert the `.tsv` files to `.jsonl` files that `inference.py` can use.

`process_raw_annotations.py` can be run like this:

```bash
python process_raw_annotations.py the_path_to_the_raw_dataset the_path_to_the_new_dataset
```

The `the_path_to_the_raw_dataset` can be either a `.tsv` file or a directory containing such files.

Run `process_raw_annotations.sh` to run commands needed for `rushifteval` and `rusemeval`:

```bash
bash process_raw_annotations.sh
```
