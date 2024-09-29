# import os
# import json
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizerFast, BertForTokenClassification
# from transformers import Trainer, TrainingArguments

# # Define your Dataset class
# class BOEDataset(Dataset):
#     def __init__(self, json_folder, tokenizer, max_length=512):
#         self.json_folder = json_folder
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.json_list = self._load_json_list()

#     def _load_json_list(self):
#         json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
#         return [os.path.join(self.json_folder, f) for f in json_files]

#     def __len__(self):
#         return len(self.json_list)

#     def __getitem__(self, idx):
#         json_file = self.json_list[idx]
#         with open(json_file, 'r') as f:
#             data = json.load(f)

#         words = []
#         labels = []
#         word_info_list = data.get('words', [])
#         if isinstance(word_info_list, list):
#             for sublist in word_info_list:
#                 if isinstance(sublist, list):
#                     for word_info in sublist:
#                         if isinstance(word_info, dict):
#                             word = word_info.get('description', '')
#                             label = word_info.get('tag', 'O')
#                             words.append(word)
#                             labels.append(label)

#         encoding = self.tokenizer(words, 
#                                   is_split_into_words=True, 
#                                   return_offsets_mapping=False, 
#                                   padding='max_length', 
#                                   truncation=True, 
#                                   max_length=self.max_length)

#         label_ids = [self.label_to_id(l) for l in labels]
#         label_ids = label_ids[:self.max_length]  # truncate to max_length
#         label_ids += [-100] * (self.max_length - len(label_ids))  # pad to max_length

#         return {
#             'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
#             'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
#             'labels': torch.tensor(label_ids, dtype=torch.long)
#         }

#     def label_to_id(self, label):
#         label_map = {"O": 0, "port-no": 1, "be-no": 2, "be-date": 3, "be-type": 4, "iecbr": 5}
#         return label_map.get(label, 0)

# # Define custom collate function
# def collate_fn(batch):
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     attention_mask = torch.stack([item['attention_mask'] for item in batch])
#     labels = torch.stack([item['labels'] for item in batch])
    
#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'labels': labels
#     }

# # Define your training function
# def train_model(json_folder, image_folder):
#     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#     dataset = BOEDataset(json_folder=json_folder, tokenizer=tokenizer)
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

#     model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=6)

#     training_args = TrainingArguments(
#         output_dir='./trained_model',
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         save_steps=10_000,
#         save_total_limit=2,
#         logging_dir='./logs',
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#     )

#     trainer.train()

# if __name__ == "__main__":
#     JSON_FOLDER = r"C:\Users\RPA_Pluto\Desktop\Tushar\BOE Vijaynagar\BERT-BOE-New\BERT\BOE_Part1_Labels_24-09-13T130413\latest"
#     IMAGE_FOLDER = r"C:\Users\RPA_Pluto\Desktop\Tushar\BOE Vijaynagar\BERT-BOE-New\BERT\BOE_Part1_Labels_24-09-13T130413\images"  # This folder is not used in this code
#     train_model(JSON_FOLDER, IMAGE_FOLDER)
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import Trainer, TrainingArguments

# Define your Dataset class
class BOEDataset(Dataset):
    def __init__(self, base_dir, tokenizer, max_length=512):
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data_from_all_parts()

    def _load_data_from_all_parts(self):
        all_data = []
        # Loop through all parts (folders in base_dir)
        for part in os.listdir(self.base_dir):
            part_dir = os.path.join(self.base_dir, part)
            if os.path.isdir(part_dir):  # Ensure it's a directory
                json_folder = os.path.join(part_dir, 'latest')  # Assuming 'latest' contains JSON files
                json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

                for json_file in json_files:
                    json_path = os.path.join(json_folder, json_file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    words = []
                    labels = []
                    word_info_list = data.get('words', [])
                    if isinstance(word_info_list, list):
                        for sublist in word_info_list:
                            if isinstance(sublist, list):
                                for word_info in sublist:
                                    if isinstance(word_info, dict):
                                        word = word_info.get('description', '')
                                        label = word_info.get('tag', 'O')
                                        words.append(word)
                                        labels.append(label)

                    all_data.append({
                        'words': words,
                        'labels': labels
                    })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        words = sample['words']
        labels = sample['labels']

        encoding = self.tokenizer(words, 
                                  is_split_into_words=True, 
                                  return_offsets_mapping=False, 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_length)

        label_ids = [self.label_to_id(l) for l in labels]
        label_ids = label_ids[:self.max_length]  # truncate to max_length
        label_ids += [-100] * (self.max_length - len(label_ids))  # pad to max_length

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def label_to_id(self, label):
        label_map = {"O": 0, "port-no": 1, "be-no": 2, "be-date": 3, "be-type": 4, "iecbr": 5}
        return label_map.get(label, 0)

# Define custom collate function
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Define your training function
def train_model(base_dir):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    dataset = BOEDataset(base_dir=base_dir, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=6)

    training_args = TrainingArguments(
        output_dir='./results/combined_parts',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\rpa_arjun\Desktop\BOE Vijaynagar\BOE\Dataset"  # The folder containing all 5 parts
    train_model(BASE_DIR)
