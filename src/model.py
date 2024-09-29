# # src/model.py
# from transformers import BertForTokenClassification, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split

# def prepare_model():
#     """
#     Prepare the BERT model for token classification.
#     """
#     model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
#     return model

# def train_model(train_texts, train_labels):
#     """
#     Train the BERT-based model on the dataset.
#     """
#     model = prepare_model()
    
#     # Convert the data into trainable tensors
#     training_args = TrainingArguments(output_dir='./models', num_train_epochs=3, per_device_train_batch_size=16)
#     trainer = Trainer(model=model, args=training_args, train_dataset=train_texts, eval_dataset=train_labels)
    
#     # Train the model
#     trainer.train()
#     return model

# def predict(model, test_texts):
#     """
#     Use the trained model to predict fields from test texts.
#     """
#     predictions = model(test_texts)
#     return predictions
import torch
from transformers import BertForTokenClassification, Trainer, TrainingArguments, BertTokenizer
from sklearn.model_selection import train_test_split

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_model():
    """
    Prepare the BERT model for token classification.
    """
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust num_labels as needed
    return model

def train_model(train_texts, train_labels):
    """
    Train the BERT-based model on the dataset.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the input texts
    encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", is_split_into_words=False)

    # Prepare the labels (ensure labels are aligned with tokenized inputs)
    labels = [label + [0] * (len(enc) - len(label)) for enc, label in zip(encodings['input_ids'], train_labels)]

    # Create the dataset
    dataset = CustomDataset(encodings, labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    # Prepare the model
    model = prepare_model()
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()
    return model


def predict(model, test_texts):
    """
    Use the trained model to predict fields from test texts.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)
    
    # Get predictions
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=2)
    return predictions
