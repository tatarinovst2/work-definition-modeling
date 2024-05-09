# Wiktionary parser

This is a parser for the russian Wiktionary XML dump.
It extracts the following information from the XML dump:

* Page ID
* Title of the page (e.g. "прыгать")
* Meanings of the word (e.g. "периодически подскакивать")
* Examples of usage for each meaning (e.g. "Девочка прыгает на дворе со скакалкой.")

Example of the output for a page:

```json lines
{
    "id": 1643, 
    "title": "прыгать",
    "definitions": 
        {
            "периодически подскакивать": 
            {
                "examples": ["Девочка прыгает на дворе со скакалкой."]
            }
        }
}
```

## How to run

### 1. Project set up

Open the Terminal or Command Prompt in the root directory of this repository.

```bash
cd path/to/work-definition-modeling
```

Activate the virtual environment.

For Windows:

```bash
python3 -m venv venv
.\venv\Scripts\activate
```

For Linux and macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Make sure you've installed the requirements.
You can do it by running

```bash
python -m pip install -r requirements.txt
```

### 2. Download the latest XML dump

You can find it
[here](https://dumps.wikimedia.org/ruwiktionary/latest/ruwiktionary-latest-pages-articles.xml.bz2).
You don't need to unpack it, the parser will do it for you.

### 3. Put it under the `wiktionary_parser/data` directory

### 4. Configure the `parsing_config.json` and `cleaning_config.json` files

You can specify the following parameters:

#### Parsing Configuration

* `templates_to_remove` - templates to remove from the text.

For example, if you want to remove the template `{{помета|...}}`,
you should add the following line: `"помета"`.

* `mappings` - each dictionary in this list has instructions on how to parse a specific template.

Template is a string that could look like this: `{{template_name|argument1|argument2|...}}`.
Each argument is separated by `|`.

For example, if you want to parse the template `{{действие|...|плавать}}`,
you should add the following dictionary:

```json lines
{
    "title_index": 0,
    "title": "действие",
    "description_indexes": [
        2
    ],
    "arguments_count": 3,
    "starting_text": "действие по значению глагола "
}
```

where `title_index` is the index of the title argument (0 in this case),

`title` is the title of the template,

`description_indexes` is the list of indexes of the description arguments,
they will be concatenated to form the text representation of the template,

`arguments_count` is the number of arguments in the template (optional),

`starting_text` is the text that will be added before the description (optional),

`ending_text` is the text that will be added after the description (optional).

#### Cleaning Configuration

* `remove_latin_in_parenthesis` - whether to remove Latin characters in parentheses.

* `remove_missed_tags` - whether to remove missed tags.

* `max_definition_character_length` - maximum allowed length of definition in characters.

* `remove_entries_without_examples` - whether to remove entries without examples.

* `throw_out_definition_markers` - definitions containing these markers will be fully removed.

* `tags_to_remove` - Tags to remove.

* `markers_for_limiting` - Markers for limiting entries.

### 5. Run the parser

Open the terminal or the command line in the root directory of the project
and run the following command:

```bash
python wiktionary_parser/run_parser.py
```

The parser will create a `definitions.jsonl` file in the `data` directory.

### 6. Cleaning the dataset

If you wish to use the created dataset for the purpose of definition modeling,
you would want to clear the dataset of entries without examples, definitions that
represent grammatical meaning instead of lexical or non-informative definitions.

For this run the following command:

```bash
python wiktionary_parser/clean_dataset.py --dataset-path path/to/the/dataset \
--output-path path/to/new/dataset
```

For example:

```bash
python wiktionary_parser/clean_dataset.py --dataset-path wiktionary_parser/data/definitions.jsonl \
--output-path wiktionary_parser/data/cleaned_definitions.jsonl
```
