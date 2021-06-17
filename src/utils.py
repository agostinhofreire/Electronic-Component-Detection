import os
import random as rd
import json
import matplotlib.pyplot as plt

def create_dataset_file(dataset_path, output_file="./fibs_data_split.json"):

    rd.seed(28)

    classes = os.listdir(dataset_path)
    classes.sort()

    dataset = {}

    #max_files = 32

    val_size = 10#abs(max_files-20)//2

    for idx, _class in enumerate(classes):

        class_path = os.path.join(dataset_path, _class)
        images = os.listdir(class_path)
        rd.shuffle(images)

        temp_sets = {
            "class_id": idx,
            "5_shot_set": [],
            "10_shot_set": [],
            "15_shot_set": [],
            "val_set": [],
            "test_set": []
            }

        temp_sets["5_shot_set"] = images[:5]
        temp_sets["10_shot_set"] = images[:10]
        temp_sets["15_shot_set"] = images[:15]
        temp_sets["val_set"] = images[15:15+val_size]
        temp_sets["test_set"] = images[15+val_size:]

        if _class not in dataset.keys():
            dataset[_class] = temp_sets



    file = open(output_file, "w+")
    json.dump(dataset, file, indent=4)
    file.close()

def plot_training(H, plotPath="./train_plot.png"):

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
