import json
import random
import os
from pathlib import Path

def format_annotation():
    #Read the original annotation file
    with open('data/annotation.json', 'r') as f:
        original_data = json.load(f)
    
    for i in original_data.keys():
        for j in original_data[i]:
            j['tag'] = 'iu'

    with open('data/annotation_formatted.json', 'w') as f:
        json.dump(original_data, f, indent=4)
    

# def format_annotation():
#     # Read the original annotation file
#     with open('data/annotation.json', 'r') as f:
#         original_data = json.load(f)
    
#     # Get all image directories
#     image_dirs = [d for d in os.listdir('data/images') if os.path.isdir(os.path.join('data/images', d))]
    
#     # Shuffle the directories for random split
#     random.shuffle(image_dirs)
    
#     # Calculate split sizes
#     total = len(image_dirs)
#     train_size = int(total * 0.8)
#     test_size = int(total * 0.1)
    
#     # Split the data
#     train_dirs = image_dirs[:train_size]
#     test_dirs = image_dirs[train_size:train_size + test_size]
#     val_dirs = image_dirs[train_size + test_size:]
    
#     # Create the formatted data structure
#     formatted_data = {
#         "train": [],
#         "test": [],
#         "val": []
#     }
    
#     # Helper function to create entry
#     def create_entry(img_id, split):
#         # Get the report from original_data
#         # The structure might be different, so we need to handle it carefully
#         report = ""
#         if isinstance(original_data, dict):
#             # If original_data is a dictionary with image IDs as keys
#             if img_id in original_data:
#                 report = original_data[img_id].get("report", "")
#         elif isinstance(original_data, list):
#             # If original_data is a list of dictionaries
#             for item in original_data:
#                 if item.get("id") == img_id:
#                     report = item.get("report", "")
#                     break
        
#         return {
#             "id": img_id,
#             "report": report,
#             "image_path": [
#                 f"{img_id}/0.png",
#                 f"{img_id}/1.png"
#             ],
#             "split": split,
#             "tag": "iu"
#         }
    
#     # Add entries to each split
#     for img_id in train_dirs:
#         formatted_data["train"].append(create_entry(img_id, "train"))
    
#     for img_id in test_dirs:
#         formatted_data["test"].append(create_entry(img_id, "test"))
    
#     for img_id in val_dirs:
#         formatted_data["val"].append(create_entry(img_id, "val"))
    
#     # Save the formatted data
#     with open('data/annotation_formatted.json', 'w') as f:
#         json.dump(formatted_data, f, indent=4)
    
#     print(f"Total images: {total}")
#     print(f"Train set: {len(formatted_data['train'])}")
#     print(f"Test set: {len(formatted_data['test'])}")
#     print(f"Validation set: {len(formatted_data['val'])}")

if __name__ == "__main__":
    format_annotation() 