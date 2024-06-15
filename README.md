# EasyOCR Card Scanner

EasyOCR Card Scanner is a Python-based tool designed to scan and process trading card images for Magic the Gathering (MTG) and Pokemon Trading Card Game (TCG). This tool leverages the EasyOCR library for optical character recognition (OCR) and integrates with Scryfall and Pokemon TCG APIs to identify and categorize the cards.

## Features

- **Independent Processing**: Supports independent processing for MTG and Pokemon cards.
- **Configurable Settings**: All settings are configurable via a `tcg.cfg` file.
- **Error Handling**: Separate error handling for MTG and Pokemon cards, respecting independent `IsFlipped` settings.
- **Logging**: Comprehensive logging to track processing steps and errors.

## Requirements

- Python 3.6+
- Required Python libraries:
  - `easyocr`
  - `Pillow`
  - `numpy`
  - `opencv-python-headless`
  - `requests`
  - `unidecode`

## Installation

1. Download the script and basic .cfg file.
  

2. Install the required Python libraries:
    ```sh
    pip install easyocr Pillow numpy opencv-python-headless requests unidecode
    ``

3. Replace `your_pokemon_tcg_api_key_here` with your actual API key for the Pokemon TCG API in the cfg file.

## Usage

1. Place the images to be processed in the designated folders:
    - MTG: `Magic the Gathering/SetName/Process`
    - Pokemon: `Pokemon/SetName/Process`

2. Run the script:
    ```sh
    python easyocr_card_scanner.py
    ```

3. The script will process the images, rename them based on the identified card names and sets, and handle errors as specified.

## Configuration

### `tcg.cfg` File

- **mtg_folder**: Path to the directory containing MTG card images.
- **pokemon_folder**: Path to the directory containing Pokemon card images.
- **logging_level**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
- **is_flipped_mtg**: Whether the MTG images are flipped (true/false).
- **is_flipped_pokemon**: Whether the Pokemon images are flipped (true/false).
- **api_key**: API key for the Pokemon TCG API.

## Logging

Logs are saved in `log.txt` in the same directory as the script. The logging level can be adjusted in the `tcg.cfg` file.

## Error Handling

Errors during processing are logged and the affected files are moved to an `Errors` folder within their respective set directories. The error checker can be run manually to attempt to resolve these errors.

## Contribution

Contributions are welcome! Please create an issue or pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the OCR functionality
- [Scryfall](https://scryfall.com/docs/api) for the card data API
