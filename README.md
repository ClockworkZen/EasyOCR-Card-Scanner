# EasyOCR Card Scanner

EasyOCR Card Scanner is a Python script designed to scan images of Magic: The Gathering cards and automatically recognize and organize them into directories based on their sets. It uses the EasyOCR library for Optical Character Recognition (OCR) and the Scryfall API for card identification.

## Features

- Scans images of Magic: The Gathering cards
- Uses EasyOCR for text recognition
- Identifies cards using the Scryfall API
- Handles errors and provides an error checker to resolve OCR issues
- Logs processing details and errors

## Requirements

- Python 3.7+
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
  - `Pillow`
  - `numpy`
  - `opencv-python`

## Setup

1. Download the script.

2. Install the required Python libraries:
    ```sh
    pip install os shutil logging re unicodedata datetime unidecode requests json easyocr Pillow numpy opencv-python
    ```

3. Create a `tcg.cfg` file in the root directory of the project with the following content:
    ```plaintext
    mtg_folder=Magic the Gathering
    logging_level=WARNING
    ```

    - `mtg_folder`: The path to the directory containing your Magic: The Gathering card images, preferably sorted by set. This can be a relative path.
    - `logging_level`: The logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`).

4. Ensure the `mtg_folder` exists and contains subfolders representing card sets with card images.

## Usage

Run the script:
`python easyocr_card_scanner.py`

the script will:

1. Start by scanning the images in the specified `mtg_folder`.
2. Attempt to recognize and identify each card.
3. Rename and organize the cards into directories based on their existing folder structure.
4. Log the processing details and any errors encountered.
5. If errors are encountered, it will prompt you to run the error checker or quit.
6. The error checker will attempt to resolve OCR issues and move unresolved erroring files to an `Errors` directory.

## Error Handling

- If OCR fails during the initial processing, the script logs the errors and prompts the user to run the error checker.
- The error checker preprocesses images to improve OCR accuracy and attempts to resolve errors.
- Unresolved errors are moved to an `Errors` directory for manual review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the OCR functionality
- [Scryfall](https://scryfall.com/docs/api) for the card data API
