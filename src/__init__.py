# src/__init__.py
# Optionally include some default imports if needed
from .data_loader import load_data_from_parts
from .ocr import apply_ocr_on_images
from .preprocessing import preprocess_text
from .model import train_model, predict
from .utils import save_json
