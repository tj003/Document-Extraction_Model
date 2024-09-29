# run.py
from src.data_loader import load_data_from_parts
from src.ocr import apply_ocr_on_images
from src.preprocessing import preprocess_text, tokenize_ocr
from src.model import train_model, predict
from src.utils import save_json
from src.config import DATASET_PATH, OUTPUT_PATH
import os

def main():
    # Load and preprocess data
    labels, split_df, schema, all_images = load_data_from_parts(DATASET_PATH)
    ocr_results = apply_ocr_on_images(split_df, images_folder=all_images)

    # Preprocess OCR results
    preprocessed_texts = [preprocess_text(ocr_text) for ocr_text in ocr_results.values()]

    # Tokenize for model input
    tokenized_inputs = [tokenize_ocr(text) for text in preprocessed_texts]

    # Train model
    model = train_model(train_texts=tokenized_inputs, train_labels=labels)

    # Make predictions
    predictions = predict(model, tokenized_inputs)

    # Save predictions to output
    save_json(predictions, os.path.join(OUTPUT_PATH, "predictions.json"))

if __name__ == "__main__":
    main()
