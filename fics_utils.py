import os
import random as rd
import json

def create_dataset_file(output_file="./fibs_data_split.json"):

    rd.seed(28)

    dataset_path = "./data_fics"

    classes = os.listdir(dataset_path)
    classes.sort()

    dataset = {}

    max_files = 32

    val_size = test_size = abs(max_files-20)//2

    for idx, _class in enumerate(classes):

        class_path = os.path.join(dataset_path, _class)
        images = os.listdir(class_path)
        rd.shuffle(images)

        temp_sets = {
            "class_id": idx,
            "5_shot_set": [],
            "10_shot_set": [],
            "20_shot_set": [],
            "val_set": [],
            "test_set": []
            }

        temp_sets["5_shot_set"] = images[:5]
        temp_sets["10_shot_set"] = images[:10]
        temp_sets["20_shot_set"] = images[:20]
        temp_sets["val_set"] = images[20:20+val_size]
        temp_sets["test_set"] = images[20+val_size:]

        if _class not in dataset.keys():
            dataset[_class] = temp_sets



    file = open(output_file, "w+")
    json.dump(dataset, file, indent=4)
    file.close()
