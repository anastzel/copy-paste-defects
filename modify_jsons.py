import os 
import json
from tqdm import tqdm

src_dir = "dataset"

input_dirs = [os.path.join(src_dir, "train"), os.path.join(src_dir, "val"), os.path.join(src_dir, "test")]

for input_dir in input_dirs:
    filenames = os.listdir(input_dir)
    for filename in tqdm(filenames, total=len(filenames)):
        if filename.endswith('json'):
            
            # Open the JSON file
            with open(os.path.join(input_dir, filename), 'r') as file:
                data = json.load(file)

            # Update the ImagePath attribute
            data["imagePath"] = filename.replace('json', 'jpg')

            # Save the updated JSON file
            with open(os.path.join(input_dir, filename), 'w') as file:
                json.dump(data, file)
