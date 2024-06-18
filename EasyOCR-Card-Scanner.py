import os
import shutil
import logging
import re
import unicodedata
from datetime import datetime
from unidecode import unidecode
import requests
import json
import easyocr
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ExifTags
import numpy as np
import cv2
import base64

class ExcludeTagsFilter(logging.Filter):
    def filter(self, record):
        return 'tag:' not in record.getMessage()

def read_config():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tcg.cfg')
    if not os.path.exists(config_file):
        print("Configuration file 'tcg.cfg' not found.")
        return None, None, None, 'warning', False, False, False, None, None, True, False, False, 1024

    mtg_folder = None
    pokemon_folder = None
    lorcana_folder = None
    logging_level = 'warning'
    is_flipped_mtg = False
    is_flipped_pokemon = False
    is_flipped_lorcana = False
    pokemon_api_key = None
    openai_api_key = None
    cleanup_mode = True
    no_prompt = False
    preprocessor_enabled = False
    size_threshold = 1024

    with open(config_file, 'r') as file:
        for line in file:
            if line.startswith("mtg_folder="):
                mtg_folder = line.split("=", 1)[1].strip()
            elif line.startswith("pokemon_folder="):
                pokemon_folder = line.split("=", 1)[1].strip()
            elif line.startswith("lorcana_folder="):
                lorcana_folder = line.split("=", 1)[1].strip()
            elif line.startswith("logging_level="):
                logging_level = line.split("=", 1)[1].strip().upper()
            elif line.startswith("is_flipped_mtg="):
                is_flipped_mtg = line.split("=", 1)[1].strip().lower() == 'true'
            elif line.startswith("is_flipped_pokemon="):
                is_flipped_pokemon = line.split("=", 1)[1].strip().lower() == 'true'
            elif line.startswith("is_flipped_lorcana="):
                is_flipped_lorcana = line.split("=", 1)[1].strip().lower() == 'true'
            elif line.startswith("pokemon_api_key="):
                pokemon_api_key = line.split("=", 1)[1].strip()
            elif line.startswith("openai_api_key="):
                openai_api_key = line.split("=", 1)[1].strip()
            elif line.startswith("cleanup_mode="):
                cleanup_mode = line.split("=", 1)[1].strip().lower() == 'true'
            elif line.startswith("no_prompt="):
                no_prompt = line.split("=", 1)[1].strip().lower() == 'true'
            elif line.startswith("preprocessor_enabled="):
                preprocessor_enabled = line.split("=", 1)[1].strip().lower() == 'true'
            elif line.startswith("size_threshold="):
                size_threshold = int(line.split("=", 1)[1].strip())

    return (mtg_folder, pokemon_folder, lorcana_folder, logging_level,
            is_flipped_mtg, is_flipped_pokemon, is_flipped_lorcana,
            pokemon_api_key, openai_api_key, cleanup_mode,
            no_prompt, preprocessor_enabled, size_threshold)

mtg_folder, pokemon_folder, lorcana_folder, logging_level, is_flipped_mtg, is_flipped_pokemon, is_flipped_lorcana, pokemon_api_key, openai_api_key, cleanup_mode, no_prompt, preprocessor_enabled, size_threshold = read_config()

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log.txt')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=getattr(logging, logging_level, logging.warning), format='%(asctime)s %(levelname)s:%(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

logger = logging.getLogger('CardScanner')
logger.setLevel(getattr(logging, logging_level, logging.warning))
logger.addFilter(ExcludeTagsFilter())

if not logger.hasHandlers():
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

reader = easyocr.Reader(['en'], gpu=True)

def log_error(message):
    logger.error(message)

def sanitize_filename(name):
    name = name.replace('&', 'and')
    nfkd_form = unicodedata.normalize('NFD', name)
    sanitized_name = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    sanitized_name = re.sub(r'[^\w\s.-]', '', sanitized_name)
    sanitized_name = re.sub(r'\s+', ' ', sanitized_name).strip()
    return sanitized_name

def exif_rebalancer(image):
    try:
        for orientation in ExifTags.tags.keys():
            if ExifTags.tags[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None:
            orientation = exif.get(orientation, None)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image

def frame_trimmer(image, crop_width, crop_height):
    img_width, img_height = image.size
    left = (img_width - crop_width) // 2
    top = int((img_height - crop_height) * 0.45)
    right = (img_width + crop_width) // 2
    bottom = int(img_height - (img_height - crop_height) * 0.55)
    return image.crop((left, top, right, bottom))

def preprocessor(image_path, size_threshold):
    try:
        image = Image.open(image_path)
        image = exif_rebalancer(image)

        img_width, img_height = image.size
        if img_width > size_threshold or img_height > size_threshold:
            image = frame_trimmer(image, size_threshold, size_threshold)

        image.save(image_path)
    except Exception as e:
        log_error(f"Error processing file {image_path}: {str(e)}")

def find_image_files(folder):
    """Recursively find all image files in the given folder."""
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def sanitize_primary_name(name):
    # Convert euro symbols to 'e'
    name = name.replace('€', 'e')
    # Remove any special characters except apostrophes
    name = re.sub(r"[^\w\s']", '', name)  # keep apostrophes
    name = re.sub(r'\d+$', '', name).strip()
    return name

def sanitize_combined_name(name):
    return name.strip()

def InvertWhiteText(image_path, threshold=220):
    image = Image.open(image_path).convert('RGB')
    gray_image = ImageOps.grayscale(image)
    gray_array = np.array(gray_image)
    mask = gray_array > threshold
    image_array = np.array(image)
    image_array[mask] = [0, 0, 0]
    result_image = Image.fromarray(image_array)
    return result_image

def process_mtg_image(image_path, output_path, save_debug=False):
    image = Image.open(image_path)
    image = image.rotate(0 if is_flipped_mtg else 180)

    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    card = cv2.bitwise_and(image_np, image_np, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    card_cropped = card[y:y+h, x:x+w]
    cropped_image_pil = Image.fromarray(card_cropped)

    top_20_percent_height = int(0.2 * h)
    top_20_percent_cropped = card_cropped[:top_20_percent_height, :]
    top_20_percent_cropped_pil = Image.fromarray(top_20_percent_cropped).convert('RGB')  # Convert to RGB mode
    top_20_percent_cropped_pil.save(output_path)

    if save_debug and logger.isEnabledFor(logging.DEBUG):
        debug_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR Debug Images")
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_image_path = os.path.join(debug_folder, f"{original_name}_OCR_debug.jpg")
        top_20_percent_cropped_pil.save(debug_image_path)
        logger.debug(f"Saved debug image to {debug_image_path}")

    return top_20_percent_cropped_pil  # Return the cropped image for further processing

def process_pokemon_image(image_path, output_path, save_debug=False):
    image = Image.open(image_path)
    image = image.rotate(0 if is_flipped_pokemon else 180)

    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    card = cv2.bitwise_and(image_np, image_np, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    card_cropped = card[y:y+h, x:x+w]
    cropped_image_pil = Image.fromarray(card_cropped)

    top_20_percent_height = int(0.2 * h)
    top_20_percent_cropped = card_cropped[:top_20_percent_height, :]
    top_20_percent_cropped_pil = Image.fromarray(top_20_percent_cropped).convert('RGB')  # Convert to RGB mode
    top_20_percent_cropped_pil.save(output_path)

    if save_debug and logger.isEnabledFor(logging.DEBUG):
        debug_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR Debug Images")
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_image_path = os.path.join(debug_folder, f"{original_name}_OCR_debug.jpg")
        top_20_percent_cropped_pil.save(debug_image_path)
        logger.debug(f"Saved debug image to {debug_image_path}")

    return top_20_percent_cropped_pil  # Return the cropped image for further processing

def process_lorcana_image(image_path, output_path, save_debug=False):
    image = Image.open(image_path)
    image = image.rotate(0 if is_flipped_lorcana else 180)

    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    card = cv2.bitwise_and(image_np, image_np, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    card_cropped = card[y:y+h, x:x+w]
    cropped_image_pil = Image.fromarray(card_cropped)

    bottom_55_percent_height = int(0.55 * h)
    bottom_55_percent_cropped = card_cropped[-bottom_55_percent_height:, :]
    bottom_55_percent_cropped_pil = Image.fromarray(bottom_55_percent_cropped)
    bottom_55_percent_cropped_pil.save(output_path)

    if save_debug and logger.isEnabledFor(logging.DEBUG):
        debug_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR Debug Images")
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_image_path = os.path.join(debug_folder, f"{original_name}_OCR_debug.jpg")
        bottom_55_percent_cropped_pil.save(debug_image_path)
        logger.debug(f"Saved debug image to {debug_image_path}")

    return bottom_55_percent_cropped_pil  # Return the cropped image for further processing

def get_mtg_card_name_and_set(image_path, process_image_function, api_url_template, fuzzy_search_function, headers=None, set_folder=None):
    temp_image_path = os.path.join(os.path.dirname(image_path), 'temp_image.jpg')
    cropped_image = process_image_function(image_path, temp_image_path, save_debug=True)
    try:
        # First OCR pass
        results = reader.readtext(temp_image_path, detail=0)
        logger.debug(f"OCR detected text: {results}")
        if results and len(results) > 0:
            card_name = sanitize_primary_name(results[0])
            if len(results) > 1:
                card_name += ' ' + sanitize_primary_name(results[1])
            logger.debug(f"Using card name for api: {card_name}")
            card_name, set_name = fuzzy_search_function(card_name.lower(), image_path, api_url_template, headers)
            if card_name and set_name:
                os.remove(temp_image_path)
                return card_name, set_name

        # Second attempt with inverted white text
        inverted_image = InvertWhiteText(temp_image_path)
        inverted_image.save(temp_image_path)

        # Save the inverted image as debug if needed
        if logger.isEnabledFor(logging.DEBUG):
            debug_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR Debug Images")
            if not os.path.exists(debug_folder):
                os.makedirs(debug_folder)
            original_name = os.path.splitext(os.path.basename(image_path))[0]
            inverted_debug_image_path = os.path.join(debug_folder, f"{original_name}_OCR_debug_INVERTED.jpg")
            inverted_image.save(inverted_debug_image_path)
            logger.debug(f"Saved inverted debug image to {inverted_debug_image_path}")

        results = reader.readtext(temp_image_path, detail=0)
        logger.debug(f"Inverted OCR detected text: {results}")
        os.remove(temp_image_path)
        if results and len(results) > 0:
            card_name = sanitize_primary_name(results[0])
            if len(results) > 1:
                card_name += ' ' + sanitize_primary_name(results[1])
            logger.debug(f"Using inverted card name for api: {card_name}")
            return fuzzy_search_function(card_name.lower(), image_path, api_url_template, headers)

        logger.debug("No valid text detected by OCR in both attempts.")
        return None, None
    except Exception as e:
        log_error(f"Failed to process image {image_path}: {str(e)}")
        return None, None

def get_lorcana_card_name_and_set(image_path, process_image_function, api_url_template, fuzzy_search_function, headers=None, set_folder=None):
    temp_image_path = os.path.join(os.path.dirname(image_path), 'temp_image.jpg')
    cropped_image = process_image_function(image_path, temp_image_path, save_debug=True)
    try:
        # First OCR pass
        results = reader.readtext(temp_image_path, detail=0)
        logger.debug(f"OCR detected text: {results}")
        if results and len(results) >= 4:
            primary_name = sanitize_primary_name(results[0])
            secondary_name = sanitize_primary_name(results[3])
            combined_name = f"{primary_name} - {secondary_name}"
            combined_name = sanitize_combined_name(combined_name)
            logger.debug(f"Using combined card name for api: {combined_name}")
            card_name, set_name = fuzzy_search_function(combined_name.lower(), image_path, api_url_template, headers)
            if card_name and set_name:
                os.remove(temp_image_path)
                return card_name, set_name
        else:
            logger.debug("Not enough text detected by OCR to form a valid card name.")
        os.remove(temp_image_path)
        return None, None
    except Exception as e:
        log_error(f"Failed to process image {image_path}: {str(e)}")
        return None, None

def get_pokemon_card_name_and_set(image_path, process_image_function, api_url_template, fuzzy_search_function, headers=None, set_name=''):
    temp_image_path = os.path.join(os.path.dirname(image_path), 'temp_image.jpg')
    cropped_image = process_image_function(image_path, temp_image_path, save_debug=True)
    temp_cropped_again_path = None
    try:
        # First OCR pass
        results = reader.readtext(temp_image_path, detail=0)
        logger.debug(f"OCR detected text: {results}")

        card_name = None

        if results and len(results) > 1:
            second_line = results[1].lower()

            if second_line == "ancient":
                # Skip the second line as it contains "Ancient"
                card_name = sanitize_primary_name(results[0])
            elif second_line in ["trainer", "energy", "traner"]:
                # Crop the image to keep the bottom 65%
                width, height = cropped_image.size
                cropped_again = cropped_image.crop((0, int(height * 0.35), width, height))
                temp_cropped_again_path = os.path.join(os.path.dirname(image_path), 'temp_cropped_again.jpg')
                cropped_again.save(temp_cropped_again_path)

                # Re-OCR on cropped image
                results = reader.readtext(temp_cropped_again_path, detail=0)
                logger.debug(f"Re-OCR detected text: {results}")

                if results and len(results) > 1:
                    card_name = f"{sanitize_primary_name(results[0])} {sanitize_primary_name(results[1])}"
                else:
                    card_name = sanitize_primary_name(results[0])

                # Debug logging for the cropped again image
                if logger.isEnabledFor(logging.DEBUG):
                    debug_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR Debug Images")
                    if not os.path.exists(debug_folder):
                        os.makedirs(debug_folder)
                    original_name = os.path.splitext(os.path.basename(image_path))[0]
                    debug_image_path = os.path.join(debug_folder, f"{original_name}_OCR_debug_CROPPED_AGAIN.jpg")
                    cropped_again.save(debug_image_path)
                    logger.debug(f"Saved cropped again debug image to {debug_image_path}")
            else:
                card_name = sanitize_primary_name(results[1])  # Use only the second detected line if it's not "Ancient"

        if card_name:
            # Strip 'ex' and 'ancient' from the card names if present
            card_name = re.sub(r'ex\b', '', card_name, flags=re.IGNORECASE).strip()
            card_name = re.sub(r'\bancient\b', '', card_name, flags=re.IGNORECASE).strip()
            logger.debug(f"Using card name for api: {card_name}")
            card_name, set_name = fuzzy_search_function(card_name.lower(), image_path, api_url_template, headers, set_name)

            if card_name and set_name:
                os.remove(temp_image_path)
                if temp_cropped_again_path and os.path.exists(temp_cropped_again_path):
                    os.remove(temp_cropped_again_path)
                return card_name, set_name

        os.remove(temp_image_path)
        if temp_cropped_again_path and os.path.exists(temp_cropped_again_path):
            os.remove(temp_cropped_again_path)
        return None, None
    except Exception as e:
        log_error(f"Failed to process image {image_path}: {str(e)}")
        if temp_cropped_again_path and os.path.exists(temp_cropped_again_path):
            os.remove(temp_cropped_again_path)
        return None, None

def fuzzy_search_card_name(card_name, image_path, api_url_template, headers=None):
    try:
        response = requests.get(api_url_template.format(card_name), headers=headers)
        if response.status_code == 200:
            card_data = response.json()
            logger.debug(f"API response: {str(card_data).encode('utf-8', errors='ignore').decode('utf-8')}")
            card_name = card_data['name']
            set_name = card_data.get('set_name', 'Unknown Set')
            logger.info(f"Identified card '{card_name}' from set '{set_name}' for image {os.path.relpath(image_path, start=mtg_folder)}")
            print(f"Card: '{card_name}' from Set: '{set_name}' was identified.")
            return card_name, set_name
        else:
            return None, None
    except Exception as e:
        return None, None

def fuzzy_search_card_name_pokemon(card_name, image_path, api_url_template, headers=None, set_name=''):
    logger.debug(f"Fuzzy searching for card name: {card_name} in set: {set_name}")
    try:
        query = f'name:"{card_name}" set.name:"{set_name}"'
        response = requests.get(api_url_template.format(query), headers=headers)
        if response.status_code == 200:
            card_data = response.json()
            logger.debug(f"api response: {str(card_data).encode('utf-8', errors='ignore').decode('utf-8')}")
            card_name = card_data['data'][0]['name']
            set_name = card_data['data'][0].get('set', {}).get('name', 'Unknown Set')
            logger.info(f"Identified card '{card_name}' from set '{set_name}' for image {os.path.relpath(image_path, start=pokemon_folder)}")
            print(f"Card: '{card_name}' from Set: '{set_name}' was identified.")
            return card_name, set_name
        else:
            logger.debug(f"api request failed with status code {response.status_code} for card name: {card_name}")
            return None, None
    except Exception as e:
        logger.debug(f"Exception during api request: {str(e)}")
        return None, None

def fuzzy_search_card_name_lorcana(card_name, image_path, api_url_template, headers=None):
    try:
        response = requests.get(api_url_template.format(card_name))
        response_json = response.json()
        logger.debug(f"Lorcana api response: {str(response_json).encode('utf-8', errors='ignore').decode('utf-8')}")

        if response.status_code == 200 and isinstance(response_json, list) and response_json:
            card_data = response_json[0]
            full_card_name = card_data.get('Name')
            set_name = card_data.get('Set_Name', 'Unknown Set')
            if full_card_name:
                logger.info(f"Identified card '{full_card_name}' from set '{set_name}' for image {os.path.relpath(image_path, start=lorcana_folder)}")
                print(f"Card: '{full_card_name}' from Set: '{set_name}' was identified.")
                return full_card_name, set_name
            else:
                return None, None
        else:
            return None, None
    except Exception as e:
        return None, None

def move_file(src, dest_dir, new_name=None):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if new_name is None:
        new_name = os.path.basename(src)

    base_name, ext = os.path.splitext(new_name)
    new_path = os.path.join(dest_dir, new_name)
    counter = 1

    while os.path.exists(new_path):
        new_name = f"{base_name}_{counter}{ext}"
        new_path = os.path.join(dest_dir, new_name)
        counter += 1

    shutil.move(src, new_path)
    return new_name

def process_directory(import_directory, process_image_function, api_url_template, fuzzy_search_function, headers=None, card_type=''):
    total_files_processed = 0
    total_errors_encountered = 0
    error_files = []
    sets_without_process = []
    notified = False

    if card_type == 'mtg':
        get_card_name_and_set_function = get_mtg_card_name_and_set
    elif card_type == 'pokemon':
        get_card_name_and_set_function = get_pokemon_card_name_and_set
    elif card_type == 'lorcana':
        get_card_name_and_set_function = get_lorcana_card_name_and_set
    else:
        get_card_name_and_set_function = None

    if not get_card_name_and_set_function:
        raise ValueError(f"Unknown card type: {card_type}")

    for set_folder in os.listdir(import_directory):
        set_path = os.path.join(import_directory, set_folder)
        process_path = os.path.join(set_path, 'Process')

        if os.path.isdir(set_path):
            if not notified:
                print(f"Accessing {import_directory} directory.")
                logger.info(f"Accessing {import_directory} directory.")
                notified = True

            print(f"Scanning set directory: {set_folder}")
            logger.info(f"Scanning set directory: {set_folder}")
            if not os.path.exists(process_path):
                sets_without_process.append(set_folder)
                print(f"No new files were found in {set_folder}/Process")
                logger.info(f"No new files were found in {set_folder}/Process")
                continue

            found_files = False
            for root, dirs, files in os.walk(process_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        found_files = True
                        total_files_processed += 1
                        file_path = os.path.join(root, file)
                        card_name, set_name = get_card_name_and_set_function(file_path, process_image_function, api_url_template, fuzzy_search_function, headers, set_folder)

                        if card_name and set_name:
                            new_name = f"{sanitize_filename(card_name)}{os.path.splitext(file)[1]}"
                            move_file(file_path, set_path, new_name)
                            relative_root = os.path.relpath(root, start=import_directory)
                            print(f"Renamed '{file}' to '{new_name}' in {relative_root}")
                            logger.info(f"Renamed '{file}' to '{new_name}' in {relative_root}")
                        else:
                            total_errors_encountered += 1
                            error_files.append(file_path)
                            move_file(file_path, os.path.join(set_path, 'Errors'))
                            log_error(f"Failed to identify card for '{os.path.relpath(file_path, start=import_directory)}'")
                    else:
                        total_errors_encountered += 1
                        error_files.append(file_path)
                        move_file(file_path, os.path.join(set_path, 'Errors'))
                        log_error(f"Failed OCR recognition for '{os.path.relpath(file_path, start=import_directory)}'")

            if not found_files:
                sets_without_process.append(set_folder)
                print(f"No new files were found in {set_folder}/Process")
                logger.info(f"No new files were found in {set_folder}/Process")

    if total_files_processed == 0:
        print("No new files found.")
        logger.info("No new files found.")
    else:
        print(f"\nTotal files processed: {total_files_processed}")
        logger.info(f"Total files processed: {total_files_processed}")

    if total_errors_encountered > 0:
        return error_files
    else:
        return []

def preprocess_image(image):
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_image = enhancer.enhance(2)
    blurred_image = enhanced_image.filter(ImageFilter.GaussianBlur(1))
    return blurred_image

def prompt_before_running_error_checker(total_errors):
    print(f"\nTotal errors encountered: {total_errors}")
    if no_prompt:
        return 'enter'
    else:
        print("Press Enter to run Error Checker or Q to quit.")
        logger.info("Waiting for user input to run Error Checker or quit.")
        user_input = input().strip().lower()
        return user_input

def error_checker_mtg(error_files, process_image_function, api_url_template, headers=None):
    resolved_errors = 0
    critical_errors = 0

    for file_path in error_files:
        try:
            # First Stage: Crop 50% from the top and perform OCR
            image = Image.open(file_path)
            image = image.rotate(0 if is_flipped_mtg else 180)
            width, height = image.size
            cropped_image = image.crop((0, 0, width, height // 2))

            results = reader.readtext(np.array(cropped_image), detail=0)
            if results:
                card_name = sanitize_primary_name(results[0])
                if len(results) > 1:
                    card_name += ' ' + sanitize_primary_name(results[1])
                logger.debug(f"First stage OCR detected text: {card_name}")
                card_name, set_name = fuzzy_search_card_name(card_name, file_path, api_url_template, headers)
                if card_name and set_name:
                    new_name = f"{sanitize_filename(card_name)} - {sanitize_filename(set_name)}{os.path.splitext(os.path.basename(file_path))[1]}"
                    move_file(file_path, os.path.dirname(os.path.dirname(file_path)), new_name)
                    relative_file_path = os.path.relpath(file_path, start=mtg_folder)
                    relative_new_path = os.path.relpath(os.path.join(os.path.dirname(os.path.dirname(file_path)), new_name), start=mtg_folder)
                    print(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    logger.info(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    resolved_errors += 1
                    continue  # Move to the next file if resolved
                else:
                    logger.debug(f"First stage verification failed for {file_path}")

            # Second Stage: Use the unedited image with preprocessing
            preprocessed_image = preprocess_image(image)
            results = reader.readtext(np.array(preprocessed_image), detail=0)
            if results:
                card_name = sanitize_primary_name(results[0])
                if len(results) > 1:
                    card_name += ' ' + sanitize_primary_name(results[1])
                logger.debug(f"Second stage OCR detected text: {card_name}")
                card_name, set_name = fuzzy_search_card_name(card_name, file_path, api_url_template, headers)
                if card_name and set_name:
                    new_name = f"{sanitize_filename(card_name)} - {sanitize_filename(set_name)}{os.path.splitext(os.path.basename(file_path))[1]}"
                    move_file(file_path, os.path.dirname(os.path.dirname(file_path)), new_name)
                    relative_file_path = os.path.relpath(file_path, start=mtg_folder)
                    relative_new_path = os.path.relpath(os.path.join(os.path.dirname(os.path.dirname(file_path)), new_name), start=mtg_folder)
                    print(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    logger.info(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    resolved_errors += 1
                else:
                    move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
                    log_error(f"Failed to identify card for '{os.path.relpath(file_path, start=mtg_folder)}' in error checker")
                    critical_errors += 1
            else:
                move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
                log_error(f"Second stage OCR failed recognition for '{os.path.relpath(file_path, start=mtg_folder)}'")
                critical_errors += 1
        except Exception as e:
            move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
            log_error(f"Error checker failed to process image {os.path.relpath(file_path, start=mtg_folder)}: {str(e)}")
            critical_errors += 1

    return critical_errors

def error_checker_pokemon(error_files, process_image_function, api_url_template, headers=None):
    resolved_errors = 0
    critical_errors = 0

    for file_path in error_files:
        try:
            image = Image.open(file_path)
            image = image.rotate(0 if is_flipped_pokemon else 180)

            preprocessed_image = preprocess_image(image)
            results = reader.readtext(np.array(preprocessed_image), detail=0)
            if results and len(results) > 1:
                card_name = sanitize_primary_name(results[1])  # Use only the second detected line
                logger.debug(f"Error checker OCR detected text: {card_name}")
                card_name, set_name = fuzzy_search_card_name_pokemon(card_name.lower(), file_path, api_url_template, headers)
                if card_name and set_name:
                    new_name = f"{sanitize_filename(card_name)} - {sanitize_filename(set_name)}{os.path.splitext(os.path.basename(file_path))[1]}"
                    move_file(file_path, os.path.dirname(os.path.dirname(file_path)), new_name)
                    relative_file_path = os.path.relpath(file_path, start=pokemon_folder)
                    relative_new_path = os.path.relpath(os.path.join(os.path.dirname(os.path.dirname(file_path)), new_name), start=pokemon_folder)
                    print(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    logger.info(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    resolved_errors += 1
                else:
                    move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
                    log_error(f"Failed to identify card for '{os.path.relpath(file_path, start=pokemon_folder)}' in error checker")
                    critical_errors += 1
            else:
                move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
                log_error(f"Error checker failed OCR recognition for '{os.path.relpath(file_path, start=pokemon_folder)}'")
                critical_errors += 1
        except Exception as e:
            move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
            log_error(f"Error checker failed to process image {os.path.relpath(file_path, start=pokemon_folder)}: {str(e)}")
            critical_errors += 1

    return critical_errors

def error_checker_lorcana(error_files, process_image_function, api_url_template, headers=None):
    resolved_errors = 0
    critical_errors = 0

    for file_path in error_files:
        try:
            image = Image.open(file_path)
            image = image.rotate(0 if is_flipped_lorcana else 180)

            preprocessed_image = preprocess_image(image)
            results = reader.readtext(np.array(preprocessed_image), detail=0)
            if results:
                primary_name = sanitize_primary_name(results[0])
                if len(results) > 3 and results[1].isdigit() and results[2].isdigit():
                    secondary_name = results[3]
                    combined_name = f"{primary_name} - {secondary_name}".lower()
                else:
                    combined_name = primary_name.lower()
                combined_name = sanitize_combined_name(combined_name)
                logger.debug(f"Error checker OCR detected text: {results}")
                card_name, set_name = fuzzy_search_card_name_lorcana(combined_name, file_path, api_url_template)
                if card_name and set_name:
                    new_name = f"{sanitize_filename(card_name)}{os.path.splitext(os.path.basename(file_path))[1]}"
                    move_file(file_path, os.path.dirname(os.path.dirname(file_path)), new_name)
                    relative_file_path = os.path.relpath(file_path, start=lorcana_folder)
                    relative_new_path = os.path.relpath(os.path.join(os.path.dirname(os.path.dirname(file_path)), new_name), start=lorcana_folder)
                    print(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    logger.info(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    resolved_errors += 1
                else:
                    move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
                    log_error(f"Failed to identify card for '{os.path.relpath(file_path, start=lorcana_folder)}' in error checker")
                    critical_errors += 1
            else:
                move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
                log_error(f"Error checker failed OCR recognition for '{os.path.relpath(file_path, start=lorcana_folder)}'")
                critical_errors += 1
        except Exception as e:
            move_file(file_path, os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Errors'))
            log_error(f"Error checker failed to process image {os.path.relpath(file_path, start=lorcana_folder)}: {str(e)}")
            critical_errors += 1

    return critical_errors

def find_error_files(base_folder):
    error_files_dict = {}
    for root, dirs, files in os.walk(base_folder):
        if 'Errors' in dirs:
            error_folder_path = os.path.join(root, 'Errors')
            error_files = [os.path.join(error_folder_path, f) for f in os.listdir(error_folder_path) if os.path.isfile(os.path.join(error_folder_path, f))]
            relative_error_folder_path = os.path.relpath(error_folder_path, start=base_folder)
            error_files_dict[relative_error_folder_path] = error_files
            print(f"Found {len(error_files)} error files in {relative_error_folder_path}")
            logger.info(f"Found {len(error_files)} error files in {relative_error_folder_path}")
    return error_files_dict

def clean_up_empty_error_directories(base_folder):
    empty_dirs_removed = False
    for root, dirs, files in os.walk(base_folder):
        if 'Errors' in dirs:
            error_folder_path = os.path.join(root, 'Errors')
            if not os.listdir(error_folder_path):
                os.rmdir(error_folder_path)
                empty_dirs_removed = True
                logger.info(f"Removed empty directory: {error_folder_path}")
    return empty_dirs_removed

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        log_error(f"Failed to encode image {image_path}: {str(e)}")
        return None

def critical_error_scanner(image_path):
    base64_image = encode_image(image_path)
    if not base64_image:
        log_error(f"Image encoding failed for {image_path}")
        return None, None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a trading card game expert that responds in json."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please identify this card. Return the card name and tcg name."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        logger.debug(f"Received response for {image_path}: {response_data}")
        try:
            content = response_data['choices'][0]['message']['content']
            content_json = json.loads(content.strip('```json\n'))
            return content_json.get("card_name"), content_json.get("tcg_name")
        except (KeyError, json.JSONDecodeError) as e:
            log_error(f"Error parsing json response for {image_path}: {str(e)}")
            return None, None
    else:
        log_error(f"API request failed for {image_path} with status code {response.status_code}")
        logger.debug(f"Response content: {response.content}")
        return None, None

def critical_error_processor(base_folder):
    error_files_dict = find_error_files(base_folder)
    remaining_critical_errors = 0

    for folder, error_files in error_files_dict.items():
        if error_files:
            print(f"\nProcessing critical errors from {len(error_files)} files in {folder}.")
            logger.info(f"Processing critical errors from {len(error_files)} files in {folder}.")
            for error_file in error_files:
                card_name, tcg_name = critical_error_scanner(error_file)
                if card_name and tcg_name:
                    sanitized_card_name = sanitize_filename(card_name)
                    dest_dir = os.path.dirname(os.path.dirname(error_file))
                    original_name = os.path.basename(error_file)
                    new_name = f"{sanitized_card_name}{os.path.splitext(original_name)[1]}"
                    new_name = move_file(error_file, dest_dir, new_name)
                    relative_original_path = os.path.relpath(error_file, start=base_folder)
                    relative_new_path = os.path.relpath(os.path.join(dest_dir, new_name), start=base_folder)
                    print(f"Renamed '{relative_original_path}' to '{relative_new_path}' in critical error processor")
                    logger.info(f"Renamed '{relative_original_path}' to '{relative_new_path}' in critical error processor")
                else:
                    log_error(f"Failed OCR recognition for '{error_file}'")
                    remaining_critical_errors += 1

    if remaining_critical_errors > 0:
        print(f"{remaining_critical_errors} number of critical errors still remain. Please ensure legibility and rotation.")
        logger.info(f"{remaining_critical_errors} number of critical errors still remain. Please ensure legibility and rotation.")

def main():
    preprocessor_run = False

    if preprocessor_enabled:
        print("Warning! The Preprocessor has been enabled! If you choose to continue, your image files will potentially be rotated and cropped prior to OCR detection if they have invalid exif data or exceed the size threshold. This is irreversible. Please ensure you have a backup before proceeding, if you're concerned.")
        choice = input("Press 'Y' to continue or 'Q' to quit: ").strip().upper()
        if choice == 'Q':
            print("Exiting program.")
            return
        elif choice == 'Y':
            for folder in [mtg_folder, pokemon_folder, lorcana_folder]:
                if folder:
                    image_files = find_image_files(folder)
                    for file_path in image_files:
                        print(f"Processing image file: {file_path}")
                        preprocessor(file_path, size_threshold)
            preprocessor_run = True
        else:
            print("Invalid choice. Exiting.")
            return

    if preprocessor_run:
        print("Preprocessor completed. Continuing to main processing.")
    else:
        print("Preprocessor is disabled. Continuing to main processing.")

    mtg_processed = False
    pokemon_processed = False
    lorcana_processed = False

    logger.info(f"mtg_folder: {mtg_folder}")
    logger.info(f"pokemon_folder: {pokemon_folder}")
    logger.info(f"lorcana_folder: {lorcana_folder}")

    mtg_errors = []
    pokemon_errors = []
    lorcana_errors = []

    if mtg_folder:
        if os.path.exists(mtg_folder):
            print("Accessing Magic the Gathering folder.")
            logger.info("Accessing Magic the Gathering folder.")
            mtg_errors = process_directory(mtg_folder, process_mtg_image, 'https://api.scryfall.com/cards/named?fuzzy={}', fuzzy_search_card_name, headers=None, card_type='mtg')
            mtg_processed = True
        else:
            print("No folder found for Magic the Gathering.")
            logger.warning("No folder found for Magic the Gathering.")

    if pokemon_folder:
        if os.path.exists(pokemon_folder):
            print("Accessing Pokemon folder.")
            logger.info("Accessing Pokemon folder.")
            pokemon_errors = process_directory(pokemon_folder, process_pokemon_image, 'https://api.pokemontcg.io/v2/cards?q={}', fuzzy_search_card_name_pokemon, headers={'X-Api-Key': pokemon_api_key}, card_type='pokemon')
            pokemon_processed = True
        else:
            print("No folder found for Pokemon.")
            logger.warning("No folder found for Pokemon.")

    if lorcana_folder:
        if os.path.exists(lorcana_folder):
            print("Accessing Lorcana folder.")
            logger.info("Accessing Lorcana folder.")
            lorcana_errors = process_directory(lorcana_folder, process_lorcana_image, 'https://api.lorcana-api.com/cards/fetch?search%3Dname~{}', fuzzy_search_card_name_lorcana, headers=None, card_type='lorcana')
            lorcana_processed = True
        else:
            print("No folder found for Lorcana.")
            logger.warning("No folder found for Lorcana.")

    if not mtg_processed and not pokemon_processed and not lorcana_processed:
        print("No folders found for processing.")
        logger.info("No folders found for processing.")

    all_errors = mtg_errors + pokemon_errors + lorcana_errors
    if all_errors:
        user_input = prompt_before_running_error_checker(len(all_errors))
        if user_input == 'q':
            print("Exiting.")
            logger.info("User chose to quit. Exiting.")
            return

    print("Now running Error Checker...")
    mtg_error_files_dict = find_error_files(mtg_folder)
    pokemon_error_files_dict = find_error_files(pokemon_folder)
    lorcana_error_files_dict = find_error_files(lorcana_folder)

    critical_errors_mtg = sum(error_checker_mtg(mtg_error_files, process_mtg_image, 'https://api.scryfall.com/cards/named?fuzzy={}', headers=None) for mtg_error_files in mtg_error_files_dict.values())
    critical_errors_pokemon = sum(error_checker_pokemon(pokemon_error_files, process_pokemon_image, 'https://api.pokemontcg.io/v2/cards?q={}', headers={'X-Api-Key': pokemon_api_key}) for pokemon_error_files in pokemon_error_files_dict.values())
    critical_errors_lorcana = sum(error_checker_lorcana(lorcana_error_files, process_lorcana_image, 'https://api.lorcana-api.com/cards/fetch?search%3Dname~{}', headers=None) for lorcana_error_files in lorcana_error_files_dict.values())

    total_critical_errors = critical_errors_mtg + critical_errors_pokemon + critical_errors_lorcana

    if total_critical_errors > 0:
        if no_prompt:
            critical_error_processor(mtg_folder)
            critical_error_processor(pokemon_folder)
            critical_error_processor(lorcana_folder)
        else:
            print(f"{total_critical_errors} number of critical errors remain unresolved. Press Enter to run critical error scanning using OpenAI or Q to quit.")
            logger.info(f"{total_critical_errors} number of critical errors remain unresolved. Press Enter to run critical error scanning using OpenAI or Q to quit.")

            user_input = input().strip().lower()
            if user_input == "q":
                print("Exiting.")
                logger.info("User chose to quit. Exiting.")
            else:
                critical_error_processor(mtg_folder)
                critical_error_processor(pokemon_folder)
                critical_error_processor(lorcana_folder)
    else:
        print("No Error files were detected.")
        logger.info("No Error files were detected.")

    if cleanup_mode:
        print("CleanUpMode enabled. Removing empty Errors directories.")
        logger.info("CleanUpMode enabled. Removing empty Errors directories.")
        if mtg_processed:
            clean_up_empty_error_directories(mtg_folder)
        if pokemon_processed:
            clean_up_empty_error_directories(pokemon_folder)
        if lorcana_processed:
            clean_up_empty_error_directories(lorcana_folder)

    logger.info("Processing complete. Press Enter to exit.")
    input("Processing complete. Press Enter to exit.")

if __name__ == "__main__":
    main()
