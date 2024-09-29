# src/utils.py
import json
import os

def save_json(data, filename):
    """
    Save the output to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def create_output_dir(output_dir):
    """
    Create the output directory if it doesn't exist.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def log_message(message):
    """
    Log a message to the console or a log file.
    """
    print(message)
