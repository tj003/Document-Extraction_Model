# # src/ocr.py
# import pytesseract
# from PIL import Image
# import os

# def ocr_image(image_path):
#     """
#     Perform OCR on a given image file.
#     """
#     try:
#         with Image.open(image_path) as image:
#             ocr_text = pytesseract.image_to_string(image)
#         return ocr_text
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return None

# def apply_ocr_on_images(split_df, images_folder):
#     """
#     Apply OCR on all images listed in the split dataframe.
#     """
#     ocr_results = {}
    
#     for _, row in split_df.iterrows():
#         image_file = row['files']
#         image_path = os.path.join(images_folder, image_file)

#         # Check if the image file exists
#         if not os.path.exists(image_path):
#             print(f"Image file not found: {image_file}")
#             continue

#         # Perform OCR
#         ocr_text = ocr_image(image_path)
        
#         if ocr_text:
#             ocr_results[image_file] = ocr_text

#     return ocr_results
# src/ocr.py
import pytesseract
from PIL import Image
import os

def ocr_image(image_path):
    """
    Perform OCR on a given image file.
    """
    try:
        with Image.open(image_path) as image:
            ocr_text = pytesseract.image_to_string(image)
        return ocr_text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def apply_ocr_on_images(images_folder, split_df):
    """
    Apply OCR on all images listed in the split dataframe.
    """
    ocr_results = {}
    
    for _, row in split_df.iterrows():
        image_file = row['files']
        
        # Check if the image file exists in the images folder
        # Assume that images_folder is a list of full paths
        image_path = None
        for img in images_folder:
            if os.path.basename(img) == image_file:  # Check just the filename
                image_path = img
                break

        # If image_path is still None, the image wasn't found
        if image_path is None:
            print(f"Image file not found: {image_file}")
            continue

        # Perform OCR
        ocr_text = ocr_image(image_path)
        
        if ocr_text:
            ocr_results[image_file] = ocr_text  # Use the filename as the key

    return ocr_results



