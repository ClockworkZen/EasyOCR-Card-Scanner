# EasyOCR Card Scanner

## Overview

The EasyOCR Card Scanner script processes images of trading cards, identifies the cards using OCR and the Scryfall API, and sorts them into organized folders. This script is designed to work specifically with Magic: The Gathering cards but can be adapted for other trading card games.

## Features

- Scans images of trading cards and identifies them using EasyOCR and the Scryfall API.
- Renames and organizes card images into set folders.
- Handles errors by moving unrecognized images to an "Errors" folder within each set folder.
- Option to run an error checker to attempt to identify and correct unrecognized images.

## Requirements

- Python 3.x
- Required Python packages:
  - os
  - shutil
  - logging
  - re
  - unicodedata
  - datetime
  - unidecode
  - requests
  - json
  - easyocr
  - Pillow
  - numpy
  - opencv-python

Install the required Python packages using the following pip command:

```bash
pip install os shutil logging re unicodedata datetime unidecode requests json easyocr pillow numpy opencv-python
```
## Configuration

Create a `tcg.cfg` file in the same directory as the script with the following content or use the defaults:

```
mtg_folder=Magic the Gathering
logging_level=DEBUG
is_flipped=False
```

- `mtg_folder`: The directory where your Magic: The Gathering card images are stored. This folder should contain subdirectories for each card set.
- `logging_level`: The logging level (e.g., DEBUG, INFO, WARNING).
- `is_flipped`: Set to `True` if your card images appear flipped, otherwise set to `False`.

## Folder Structure

The script expects the following folder structure:
```
Script Directory
└── Magic the Gathering
├── Set Folder 1
│ └── Process
├── Set Folder 2
│ └── Process
└── Set Folder 3
└── Process
```

- Place the card images you want to process in the `Process` folder of the respective set folder.

## Usage

1. Ensure the folder structure is correct and the card images are placed in the appropriate `Process` folders.
2. Run the script:

`python easyocr_card_scanner.py`


3. The script will scan the `Process` folders, identify and rename the card images, and move them to the set folder.
4. If any errors occur, the script will move the unrecognized images to an `Errors` folder within the respective set folder.
5. If the card images appear flipped or OCR always fails, set `is_flipped=True` in the `tcg.cfg` file to correct the orientation.

## Error Handling

If errors occur, the script will:
- Log the errors in `log.txt`.
- Move the unrecognized images to an `Errors` folder within the respective set folder.
- Prompt you to run the error checker to attempt to identify and correct unrecognized images.

After running the error checker, any remaining errors will be logged and the unrecognized images will remain in the `Errors` folder for further review.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the OCR functionality
- [Scryfall](https://scryfall.com/docs/api) for the card data API
