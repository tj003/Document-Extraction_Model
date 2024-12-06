BOE Document Extraction Using BERT Model

This repository contains the implementation of a machine learning model designed to extract specific fields from Bill of Entry (BOE) documents using the BERT model. The model is fine-tuned on a custom dataset tailored for BOE document extraction tasks.

Introduction
Bill of Entry (BOE) documents are critical in international trade and customs processes, containing structured and semi-structured data. This repository focuses on using a fine-tuned BERT model to extract fields like Importer Name, HS Code, Invoice Number, Custom Duty, and more from BOE documents.

Features
Custom Dataset Support: Designed for domain-specific BOE document data.
Preprocessing Pipeline: Includes text cleaning, OCR integration, and dynamic field mapping.
High Accuracy Extraction: Uses BERT’s robust language understanding for precise field identification.
Extensibility: Modular design to accommodate additional fields or custom document types.
Dataset
The model is trained and evaluated on a custom-labeled BOE dataset. Each dataset entry includes:

Images: Scanned BOE document images.
Labels: JSON annotations specifying the target fields and values.

dataset/
|-- images/
|-- latest/
    |-- annotations.json
|-- schema.json
|-- split.csv

split.csv specifies training and validation splits.
annotations.json contains field mappings for labeled data.
Model Architecture
This project leverages a BERT-based model with the following customizations:

Fine-tuning on token-level classification for BOE fields.
Custom preprocessing to handle OCR text input.
Loss functions optimized for field-wise accuracy.

mail on tusharj071@gmail.com for more query

Feel free to contribute to this repository and improve the BOE document extraction model! For questions or suggestions, please open an issue or submit a pull request.
