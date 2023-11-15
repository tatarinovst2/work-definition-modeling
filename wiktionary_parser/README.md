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

Make sure you've installed the requirements.
You can do it by running

```bash
pip3 install -r requirements.txt
```

### 2. Download the latest XML dump

You can find it
[here](https://dumps.wikimedia.org/ruwiktionary/latest/ruwiktionary-latest-pages-articles.xml.bz2).
You don't need to unpack it, the parser will do it for you.

### 3. Put it under the `wiktionary_parser/data` directory

### 4. Configure the `wiktionary_parser_config.py` file

You can specify the following parameters:

* `templates_to_remove`- templates to remove from the text.

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

### 5. Run the parser

Open the terminal or the command line in the root directory of the project
and run the following command:

```bash
python3 -m wiktionary_parser.run_parser
```

The parser will create a `definitions.jsonl` file in the `data` directory.
