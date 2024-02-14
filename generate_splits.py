import os
import numpy as np
import json 

"""
Opens bundle files for train and test splits, writing the take_name -> formatted like this:
sfu_cooking_011_1
fair_cooking_06_6
sfu_cooking_005_6
sfu_cooking032_4
"""

# load paths
CLI_OUTPUT_DIR = "/local/juro4948/data/egoexo4d/egoexo" # Replace with the full path to the --output_directory you pass to the cli
ANNOTATIONS_PATH = os.path.join(CLI_OUTPUT_DIR, "annotations")
# See raw annotations in this dictionary
keystep_anns = json.load(open(os.path.join(ANNOTATIONS_PATH, "keystep_train.json")))
keystep_anns_val = json.load(open(os.path.join(ANNOTATIONS_PATH, "keystep_val.json")))
anns = keystep_anns["annotations"]
anns_test = keystep_anns_val["annotations"]

# Add anns_test to anns dictionary
anns.update(anns_test)

def get_take_id_from_name(take_name):
    for take_id in anns.keys():
        if anns[take_id]['take_name'] == take_name:
            return take_id
    return None

def get_take_name_from_id(take_id):
    if take_id in anns.keys():
        return anns[take_id]['take_name']
    else:
        return None

# Load the array of video names
files = np.array(os.listdir('/local/juro4948/data/egoexo4d/egoexo/features/omnivore_video'))

# get unique take ids since there are aria and multiple gopro streams for each take
# assume there are both aria and gpro data for each take (otherwise the splits won't be quite equal)
unique_take_names = set()
for fn in files:
    take_id = fn.split('_')[0]
    take_name = get_take_name_from_id(take_id)
    if take_name not in unique_take_names and take_name is not None:
        unique_take_names.add(take_name)

files = np.array(list(unique_take_names))


# Define the path to save the splits in bundle files
# splits_dir = '/home/juro4948/gravit/data/egoexo4d/egoexo4d/preprocessed_old/GraVi-T-rawfn_splits' #f"/home/juro4948/gravit/GraVi-T/data/annotations/egoexo-omnivore/splits"
splits_dir = '/home/juro4948/gravit/GraVi-T/data/annotations/egoexo-omnivore/splits'
if not os.path.exists(splits_dir):
    os.makedirs(splits_dir)

# Split the files into 5 equal-sized splits
splits = np.array_split(files, 5)

# Save the splits as the test splits
for i, split in enumerate(splits):
    split_file_path = os.path.join(splits_dir, f'test.split{i+1}.bundle')

    with open(split_file_path, "w") as split_file:
        split_file.write("\n".join(split))
    
    print(f"Number of lines in {split_file_path}: {len(split)}")

# Now save train splits (all files not in the test split are in the corresponding train split)
for i, split in enumerate(splits):
    test_split = files[~np.isin(files, split)]
    split_file_path = os.path.join(splits_dir, f'train.split{i+1}.bundle')
    with open(split_file_path, "w") as split_file:
        split_file.write("\n".join(test_split))

    print(f"Number of lines in {split_file_path}: {len(test_split)}")


