# src/preprocessing.py
import difflib
from transformers import BertTokenizer

def preprocess_text(text):
    """
    Preprocess OCR text for better tokenization and matching.
    """
    # Example of simple preprocessing: remove extra spaces and lower the case
    text = text.strip().lower()
    return text

def match_fields(ocr_text, labeled_fields):
    """
    Match the OCR text with labeled fields using fuzzy matching.
    """
    matched_fields = {}
    for field_name, field_value in labeled_fields.items():
        closest_match = difflib.get_close_matches(field_value, [ocr_text], n=1, cutoff=0.5)
        matched_fields[field_name] = closest_match[0] if closest_match else None
    return matched_fields

def tokenize_ocr(ocr_text):
    """
    Tokenize OCR text using a pretrained BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(ocr_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    return tokens
