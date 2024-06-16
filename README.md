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
- Can use OpenAI API for critical error scanning.

## Configuration
The tool uses a configuration file `tcg.cfg` to set up directories and options. Below are the options supported in the configuration file:

- **mtg_folder**: Directory path where Magic: The Gathering card images are stored.
- **pokemon_folder**: Directory path where Pokemon card images are stored.
- **lorcana_folder**: Directory path where Lorcana card images are stored.
- **logging_level**: Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is `WARNING`.
- **is_flipped_mtg**: Set to `true` if MTG card images are flipped; otherwise, `false`. Default is `false`.
- **is_flipped_pokemon**: Set to `true` if Pokemon card images are flipped; otherwise, `false`. Default is `false`.
- **is_flipped_lorcana**: Set to `true` if Lorcana card images are flipped; otherwise, `false`. Default is `false`.
- **pokemon_api_key**: API key for accessing the Pokemon TCG API.
- **openai_api_key**: API key for accessing the OpenAI API for critical error scanning.
- **CleanUpMode**: Set to `true` to enable cleanup of empty error directories; otherwise, `false`. Default is `true`.

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

## Requirements
- Python 3.6+
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [OpenCV](https://opencv.org/)
- [requests](https://docs.python-requests.org/en/master/)
- [unidecode](https://pypi.org/project/Unidecode/)
- [base64](https://docs.python.org/3/library/base64.html)


To install the required modules, use:
`pip install unidecode requests easyocr Pillow numpy opencv-python`

### Running the Tool

`python easyocr_card_scanner.py`

## Obtaining API Keys

### OpenAI API Key
1. Sign up for an account at [OpenAI](https://www.openai.com/).
2. Navigate to the API section and generate a new API key.
3. Copy the API key and add it to the `openai_api_key` setting in your `tcg.cfg` file.

### Pokemon TCG API Key
1. Sign up for an account at [Pokemon TCG API](https://pokemontcg.io/).
2. Navigate to the API section and generate a new API key.
3. Copy the API key and add it to the `pokemon_api_key` setting in your `tcg.cfg` file.


## Logging
The tool logs its activities in `log.txt` located in the same directory as the script. The logging level can be configured in the `tcg.cfg` file.

## Troubleshooting
If the tool fails to recognize or identify cards, check the `log.txt` for detailed error messages. You can also enable `DEBUG` logging in the `tcg.cfg` for more verbose output.

## Acknowledgements
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for providing the OCR capabilities used in this tool.
- [Scryfall](https://scryfall.com/docs/api) for the Magic: The Gathering card API.
- [Pokémon TCG API](https://dev.pokemontcg.io/) for the Pokémon card API.
- [Lorcana API](https://lorcana-api.com/) for the Lorcana card API.
- [OpenAI](https://www.openai.com/) for use of the GPT-4o vision API.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
