print("EasyOCR Card Scanner is starting up...")

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
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

class ExcludeTagsFilter(logging.Filter):
    def filter(self, record):
        return 'tag:' not in record.getMessage()

def read_config():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tcg.cfg')
    if not os.path.exists(config_file):
        print("Configuration file 'tcg.cfg' not found.")
        return None, 'WARNING', False
    
    mtg_folder = None
    logging_level = 'WARNING'
    is_flipped = False
    
    with open(config_file, 'r') as file:
        for line in file:
            if line.startswith("mtg_folder="):
                mtg_folder = line.split("=", 1)[1].strip()
            elif line.startswith("logging_level="):
                logging_level = line.split("=", 1)[1].strip().upper()
            elif line.startswith("is_flipped="):
                is_flipped = line.split("=", 1)[1].strip().lower() == 'true'
    
    if mtg_folder and not os.path.isabs(mtg_folder):
        mtg_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), mtg_folder)

    return mtg_folder, logging_level, is_flipped

MTG_FOLDER, LOGGING_LEVEL, IS_FLIPPED = read_config()

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log.txt')
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL, logging.WARNING), format='%(asctime)s %(levelname)s:%(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

logger = logging.getLogger()
logger.addFilter(ExcludeTagsFilter())

reader = easyocr.Reader(['en'], gpu=False)

def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Error occurred. Check log.txt for details.")
    logging.error(message)

def sanitize_filename(name):
    name = name.replace('&', 'and')
    nfkd_form = unicodedata.normalize('NFD', name)
    sanitized_name = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    sanitized_name = re.sub(r'[^\w\s.-]', '', sanitized_name)
    sanitized_name = re.sub(r'\s+', ' ', sanitized_name).strip()
    return sanitized_name

def process_image(image_path, output_path, save_debug=False):
    image = Image.open(image_path)
    image = image.rotate(0 if IS_FLIPPED else 180)
    
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
    top_20_percent_cropped_pil = Image.fromarray(top_20_percent_cropped)
    top_20_percent_cropped_pil.save(output_path)
    
    if save_debug and logging.getLogger().isEnabledFor(logging.DEBUG):
        debug_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR DEBUG.jpg")
        count = 1
        while os.path.exists(debug_image_path):
            debug_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"OCR DEBUG_{count}.jpg")
            count += 1
        top_20_percent_cropped_pil.save(debug_image_path)
        logging.debug(f"Saved debug image to {debug_image_path}")

def get_card_name_and_set(image_path):
    temp_image_path = os.path.join(os.path.dirname(image_path), 'temp_image.jpg')
    process_image(image_path, temp_image_path, save_debug=True)
    try:
        results = reader.readtext(temp_image_path, detail=0)
        os.remove(temp_image_path)
        if results:
            first_line = results[0]
            logging.debug(f"OCR detected text: {first_line}")
            return first_line
        logging.debug("No text detected by OCR.")
        return None
    except Exception as e:
        log_error(f"Failed to process image {image_path}: {str(e)}")
        return None

def fuzzy_search_card_name(card_name, image_path):
    try:
        response = requests.get(f'https://api.scryfall.com/cards/named?fuzzy={card_name}')
        if response.status_code == 200:
            card_data = response.json()
            card_name = card_data['name']
            set_name = card_data.get('set_name', 'Unknown Set')
            logging.info(f"Identified card '{card_name}' from set '{set_name}' for image {os.path.relpath(image_path, start=MTG_FOLDER)}")
            return card_name, set_name
        else:
            logging.warning(f"Card not found for text: {card_name} in image {os.path.relpath(image_path, start=MTG_FOLDER)}")
            return None, None
    except Exception as e:
        logging.error(f"Error processing image {os.path.relpath(image_path, start=MTG_FOLDER)}: {e}")
        return None, None

def move_file(src, dest_dir, new_name=None):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    if new_name is None:
        new_name = os.path.basename(src)
    
    new_path = os.path.join(dest_dir, new_name)
    shutil.move(src, new_path)
    return new_name

def process_directory(import_directory):
    total_files_processed = 0
    total_errors_encountered = 0
    error_files = []
    sets_without_process = []

    for set_folder in os.listdir(import_directory):
        set_path = os.path.join(import_directory, set_folder)
        process_path = os.path.join(set_path, 'Process')
        
        if os.path.isdir(set_path):
            print(f"Scanning set directory: {set_folder}")
            if not os.path.exists(process_path):
                sets_without_process.append(set_folder)
                print(f"No new files were found in {set_folder}/Process")
                continue

            found_files = False
            for root, dirs, files in os.walk(process_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        found_files = True
                        total_files_processed += 1
                        file_path = os.path.join(root, file)
                        card_name = get_card_name_and_set(file_path)

                        if card_name:
                            card_name, set_name = fuzzy_search_card_name(card_name, file_path)
                            if card_name and set_name:
                                new_name = f"{sanitize_filename(card_name)} - {sanitize_filename(set_name)}{os.path.splitext(file)[1]}"
                                new_name = move_file(file_path, set_path, new_name)
                                relative_root = os.path.relpath(root, start=MTG_FOLDER)
                                print(f"Renamed '{file}' to '{new_name}' in {relative_root}")
                                logging.info(f"Renamed '{file}' to '{new_name}' in {relative_root}")
                            else:
                                total_errors_encountered += 1
                                error_files.append(file_path)
                                log_error(f"Failed to identify card for '{os.path.relpath(file_path, start=MTG_FOLDER)}'")
                        else:
                            total_errors_encountered += 1
                            error_files.append(file_path)
                            log_error(f"Failed OCR recognition for '{os.path.relpath(file_path, start=MTG_FOLDER)}'")

            if not found_files:
                sets_without_process.append(set_folder)
                print(f"No new files were found in {set_folder}/Process")

    if total_files_processed == 0:
        print("No new files found.")
        logging.info("No new files found.")
    else:
        print(f"\nTotal files processed: {total_files_processed}")
        logging.info(f"Total files processed: {total_files_processed}")
    
    if total_errors_encountered > 0:
        print(f"Total errors encountered: {total_errors_encountered}")
        logging.info(f"Total errors encountered: {total_errors_encountered}")
        print("Errors were found when trying to process one or more files. Press enter to run Error-checker or Q to quit.")
        
        user_input = input().strip().lower()
        if user_input == "q":
            print("Exiting.")
            logging.info("User chose to quit. Exiting.")
        else:
            error_checker(error_files)
    else:
        print("Error-checker will be skipped.")
        logging.info("Error-checker will be skipped.")

def preprocess_image(image):
    # Convert image to grayscale
    gray = image.convert('L')
    # Increase contrast
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_image = enhancer.enhance(2)
    # Apply a slight blur
    blurred_image = enhanced_image.filter(ImageFilter.GaussianBlur(1))
    return blurred_image

def error_checker(error_files):
    print("Starting error checker...")
    logging.info("Starting error checker...")

    resolved_errors = 0
    
    for file_path in error_files:
        set_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        set_dir = os.path.join(MTG_FOLDER, set_folder)
        error_dir = os.path.join(set_dir, "Errors")
        
        try:
            image = Image.open(file_path)
            image = image.rotate(0 if IS_FLIPPED else 180)
            
            preprocessed_image = preprocess_image(image)
            results = reader.readtext(np.array(preprocessed_image), detail=0)
            if results:
                first_line = results[0]
                logging.debug(f"Error checker OCR detected text: {first_line}")
                card_name, set_name = fuzzy_search_card_name(first_line, file_path)
                if card_name and set_name:
                    new_name = f"{sanitize_filename(card_name)} - {sanitize_filename(set_name)}{os.path.splitext(os.path.basename(file_path))[1]}"
                    move_file(file_path, set_dir, new_name)
                    relative_file_path = os.path.relpath(file_path, start=MTG_FOLDER)
                    relative_new_path = os.path.relpath(os.path.join(set_dir, new_name), start=MTG_FOLDER)
                    print(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    logging.info(f"Renamed '{relative_file_path}' to '{relative_new_path}' in error checker")
                    resolved_errors += 1
                else:
                    move_file(file_path, error_dir)
                    log_error(f"Failed to identify card for '{os.path.relpath(file_path, start=MTG_FOLDER)}' in error checker")
            else:
                move_file(file_path, error_dir)
                log_error(f"Error checker failed OCR recognition for '{os.path.relpath(file_path, start=MTG_FOLDER)}'")
        except Exception as e:
            move_file(file_path, error_dir)
            log_error(f"Error checker failed to process image {os.path.relpath(file_path, start=MTG_FOLDER)}: {str(e)}")

    remaining_errors = len(error_files) - resolved_errors
    if remaining_errors > 0:
        print(f"Errors were found that could not be automatically resolved. Please check the quality and rotation of the images and try again.")
        logging.info(f"Errors were found that could not be automatically resolved. Please check the quality and rotation of the images and try again.")
        print(f"Erroring files have been moved to: {os.path.abspath(error_dir)}")
        logging.info(f"Erroring files have been moved to: {os.path.abspath(error_dir)}")
    else:
        print("No more errors!")
        logging.info("No more errors!")

if __name__ == "__main__":
    if not MTG_FOLDER or not os.path.exists(MTG_FOLDER):
        print(f"MTG folder not found at location specified in tcg.cfg: {MTG_FOLDER}")
        input("Press Enter to exit.")
        exit(1)
    
    import_directory = MTG_FOLDER
    process_directory(import_directory)
    logging.info("Processing complete. Press Enter to exit.")
    input("Processing complete. Press Enter to exit.")
