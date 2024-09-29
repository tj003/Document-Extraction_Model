import os
import json
import pandas as pd
from pathlib import Path

def extract_filename(data):
    if 'be-no' in data:
        return f"{data['be-no']}.json"  # Use 'be-no' to create the filename
    return None

# Paths to the parts
parts = [
    'D:\\BOE\\ModelCode\\data\\part1',
    'D:\\BOE\\ModelCode\\data\\part2',
    'D:\\BOE\\ModelCode\\data\\part3',
    'D:\\BOE\\ModelCode\\data\\part4',
    'D:\\BOE\\ModelCode\\data\\part5'
]

# Initialize a list to store combined data
combined_data = []
schema_list = []
split_data = []

# Loop through each part
for part in parts:
    # Load schema.json
    with open(os.path.join(part, 'schema.json'), 'r') as schema_file:
        schema = json.load(schema_file)
        schema_list.append(schema)

    # Load split.csv
    split_df = pd.read_csv(os.path.join(part, 'split.csv'), names=['file', 'subset'])
    split_data.append(split_df)

    # Load JSON files from the latest folder
    latest_folder = os.path.join(part, 'latest')
    for json_file in os.listdir(latest_folder):
        if json_file.endswith('.json'):
            with open(os.path.join(latest_folder, json_file), 'r') as f:
                data = json.load(f)
                combined_data.append(data)

# Combine all split DataFrames
combined_split_df = pd.concat(split_data, ignore_index=True)

# Normalize combined JSON data
df = pd.json_normalize(combined_data)

# Extract filename from the JSON data using 'be-no' as the primary key
df['file'] = df['fields.be-no'].apply(lambda x: extract_filename(x) if pd.notnull(x) else None)

# Merge with combined split data on 'file'
df = df.merge(combined_split_df, on='file', how='left')

# Preprocessing steps (handle missing values, type conversions, etc.)
df.fillna('', inplace=True)  # Fill missing values

# Optionally, save the schema for reference
with open('combined_schema.json', 'w') as outfile:
    json.dump(schema_list, outfile)

# Export the processed data for model training
df.to_csv('combined_preprocessed_data.csv', index=False)

print("Data processing complete. Combined data saved to 'combined_preprocessed_data.csv' and schema saved to 'combined_schema.json'.")
