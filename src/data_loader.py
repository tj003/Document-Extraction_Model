# # src/data_loader.py
# import os
# import pandas as pd
# import json

# def load_json_labels(latest_folder):
#     """
#     Load labeled JSON files containing field annotations.
#     """
#     labels = {}
#     for json_file in os.listdir(latest_folder):
#         with open(os.path.join(latest_folder, json_file), 'r') as f:
#             labels[json_file] = json.load(f)
#     return labels

# def load_split_file(split_file):
#     """
#     Load the train/validate split information from split.csv.
#     Correctly split the combined 'files' and 'subset' column into two separate columns.
#     """
#     # Load the CSV file
#     df = pd.read_csv(split_file, sep='\t', engine='python')  # Assuming tab as the delimiter, adjust if necessary

#     # If the 'files' and 'subset' are in the same column, split them
#     if df.columns[0] == 'files	subset':  # Column name has an embedded tab
#         df = pd.read_csv(split_file, sep=' ', engine='python', header=None)
#         df.columns = ['files', 'subset']  # Manually set column names

#     return df

# def load_schema(schema_file):
#     """
#     Load the schema.json file which defines the structure of the labels.
#     """
#     with open(schema_file, 'r') as f:
#         schema = json.load(f)
#     return schema

# def load_data_from_parts(dataset_path):
#     """
#     Load all parts of the dataset and combine them into a single dataset.
#     """
#     all_labels = {}
#     all_splits = pd.DataFrame()

#     for part in os.listdir(dataset_path):
#         part_path = os.path.join(dataset_path, part)
#         if os.path.isdir(part_path):
#             latest_folder = os.path.join(part_path, "Latest")
#             split_file = os.path.join(part_path, "split.csv")
#             schema_file = os.path.join(part_path, "schema.json")

#             # Load labels, split, and schema for this part
#             labels = load_json_labels(latest_folder)
#             split_df = load_split_file(split_file)
#             schema = load_schema(schema_file)

#             # Combine results
#             all_labels.update(labels)
#             all_splits = pd.concat([all_splits, split_df], ignore_index=True)

#     return all_labels, all_splits, schema
# src/data_loader.py
import os
import pandas as pd
import json

def load_json_labels(latest_folder, schema_path):
    """
    Load labeled JSON files and process fields according to the schema.
    """
    # Load schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Create a map for field post-processing
    field_post_process_map = {field['name']: field.get('post_processing', None) for field in schema['extraction']}

    labels = {}
    for json_file in os.listdir(latest_folder):
        json_path = os.path.join(latest_folder, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                # Extract and process fields according to schema
                processed_fields = {}
                for field_name, field_value in data["fields"].items():
                    # Apply post-processing if defined in the schema
                    if field_name in field_post_process_map:
                        if field_post_process_map[field_name] == 'first_span':
                            processed_fields[field_name] = field_value.split()[0] if field_value else ""
                        else:
                            processed_fields[field_name] = field_value
                    else:
                        processed_fields[field_name] = field_value
                
                labels[json_file] = processed_fields
        else:
            print(f"JSON file not found: {json_path}")
    
    return labels

def load_split_file(split_file):
    """
    Load the train/validate split information from split.csv.
    """
    df = pd.read_csv(split_file, sep='\t', engine='python')

    # If the 'files' and 'subset' are in the same column, split them
    if df.columns[0] == 'files	subset':
        df = pd.read_csv(split_file, sep=' ', engine='python', header=None)
        df.columns = ['files', 'subset']

    return df

def load_schema(schema_file):
    """
    Load the schema.json file which defines the structure of the labels.
    """
    if os.path.exists(schema_file):  # Check if schema file exists
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        return schema
    else:
        print(f"Schema file not found: {schema_file}")  # Log missing file
        return {}

def load_images(images_folder):
    """
    Load images from the specified folder.
    """
    images = []
    for image_file in os.listdir(images_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions as necessary
            images.append(os.path.join(images_folder, image_file))
    return images

def load_data_from_parts(dataset_path):
    """
    Load all parts of the dataset and combine them into a single dataset.
    """
    all_labels = {}
    all_splits = pd.DataFrame()
    all_images = []

    for part in os.listdir(dataset_path):
        part_path = os.path.join(dataset_path, part)
        if os.path.isdir(part_path):
            latest_folder = os.path.join(part_path, "Latest")
            split_file = os.path.join(part_path, "split.csv")
            schema_file = os.path.join(part_path, "schema.json")
            images_folder = os.path.join(part_path, "Images")

            # Load labels, split, and schema for this part
            labels = load_json_labels(latest_folder)
            split_df = load_split_file(split_file)
            schema = load_schema(schema_file)
            images = load_images(images_folder)  # Collect images from the Images folder

            # Combine results
            all_labels.update(labels)
            all_splits = pd.concat([all_splits, split_df], ignore_index=True)
            all_images.extend(images)  # Extend the list with images from this part

    return all_labels, all_splits, schema, all_images

