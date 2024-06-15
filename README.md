# EasyOCR Card Scanner

EasyOCR Card Scanner is a tool to automatically recognize and rename TCG images using OCR and relevant APIs.

## Supported Card Games
- Magic: The Gathering
- Pokémon
- Lorcana

## Features
- Automatically recognizes card names using OCR.
- Identifies card details by querying relevant APIs.
- Renames card image files based on identified card names.
- Logs processing details for error checking.

## Configuration
The tool uses a configuration file `tcg.cfg` to set up directories and options. Below are the options supported in the configuration file:

- `mtg_folder`: Path to the directory containing Magic: The Gathering card images.
- `pokemon_folder`: Path to the directory containing Pokémon card images.
- `lorcana_folder`: Path to the directory containing Lorcana card images.
- `logging_level`: Logging level. Possible values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- `is_flipped_mtg`: Set to `true` if Magic: The Gathering images are flipped.
- `is_flipped_pokemon`: Set to `true` if Pokémon images are flipped.
- `is_flipped_lorcana`: Set to `true` if Lorcana images are flipped.
- `api_key`: API key for Pokémon TCG API. Obtain a free key from [Pokémon TCG Developer Portal](https://dev.pokemontcg.io/).

## Directory Structure
The expected directory structure for the card images is as follows:

```
├── mtg_folder
│   ├── Set1
│   │   └── Process
│   ├── Set2
│   │   └── Process
│   └── ...
├── pokemon_folder
│   ├── Set1
│   │   └── Process
│   ├── Set2
│   │   └── Process
│   └── ...
└── lorcana_folder
    ├── Set1
    │   └── Process
    ├── Set2
    │   └── Process
    └── ...
```

## Getting Started

### Prerequisites
- Python 3.6 or higher
- The following Python libraries:
  - `os`
  - `shutil`
  - `logging`
  - `re`
  - `unicodedata`
  - `datetime`
  - `unidecode`
  - `requests`
  - `json`
  - `easyocr`
  - `PIL`
  - `numpy`
  - `opencv-python`


To install the required modules, use:
`pip install unidecode requests easyocr Pillow numpy opencv-python`

### Running the Tool

`python easyocr_card_scanner.py`

### Obtaining a Pokémon TCG API Key
Visit the [Pokémon TCG Developer Portal](https://dev.pokemontcg.io/) to sign up and get a free API key.

## Logging
The tool logs its activities in `log.txt` located in the same directory as the script. The logging level can be configured in the `tcg.cfg` file.

## Troubleshooting
If the tool fails to recognize or identify cards, check the `log.txt` for detailed error messages. You can also enable `DEBUG` logging in the `tcg.cfg` for more verbose output.

## Acknowledgements
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for providing the OCR capabilities used in this tool.
- [Scryfall](https://scryfall.com/docs/api) for the Magic: The Gathering card API.
- [Pokémon TCG API](https://dev.pokemontcg.io/) for the Pokémon card API.
- [Lorcana API](https://lorcana-api.com/) for the Lorcana card API.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
