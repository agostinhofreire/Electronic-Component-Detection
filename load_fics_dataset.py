import os
import cv2
from fics_utils import create_dataset_file
import sys
import json
import numpy as np
import random as rd
import math
from itertools import combinations_with_replacement
from sklearn import utils

rd.seed(28)

class LoadFics:

    def __init__(self, train_size, dataset_path):  # 5, 10, 20

        if train_size not in [5, 10, 20]:
            print("Please choose a valid train_size -> [5, 10, 20]")
            sys.exit()

        if not os.path.exists("./fibs_data_split.json"):
            create_dataset_file(dataset_path)

        file = open("./fibs_data_split.json", "r")
        self.dataset = json.load(file)
        file.close()

        mode = {
            5: "5_shot_set",
            10: "10_shot_set",
            20: "20_shot_set"
        }

        self.train_mode = mode[train_size]

        self.dataset_path = dataset_path

    def __get_image(self, path):

        img = cv2.imread(path, 1) / 255.
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        return img

    def __get_infos(self):

        x_train = []
        y_train = []

        x_val = []
        y_val = []

        for _class in self.dataset.keys():

            for img_train in self.dataset[_class][self.train_mode]:
                path_train = os.path.join(self.dataset_path, _class, img_train)
                img = self.__get_image(path_train)

                x_train.append(img)
                y_train.append(self.dataset[_class]["class_id"])

            for img_val in self.dataset[_class][self.train_mode]:
                path_val = os.path.join(self.dataset_path, _class, img_val)
                img = self.__get_image(path_val)

                x_val.append(img)
                y_val.append(self.dataset[_class]["class_id"])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        return x_train, y_train, x_val, y_val


    def __make_pairs(self, images, labels):

        pairImages = []
        pairLabels = []

        numClasses = len(np.unique(labels))
        indexes_class = [np.where(labels == i)[0] for i in range(0, numClasses)]
        for i, class_elements in enumerate(indexes_class):
            comb = combinations_with_replacement(class_elements,
                                                 2)  # combine with replacement all elements of the same class
            qtde_same_image = 0
            for pair in list(comb):
                pairImages.append([images[pair[0]], images[pair[1]]])
                pairLabels.append([1])
                qtde_same_image = qtde_same_image + 1

            qde_min_pair = math.floor(qtde_same_image / (numClasses - 1))
            qde_max_pair = math.ceil(qtde_same_image / (numClasses - 1))
            for j in range(i + 1, numClasses):
                qtde_pair = rd.randint(qde_min_pair, qde_max_pair)
                other_class_elements = rd.sample(list(indexes_class[j]), qtde_pair)
                current_class_elements = rd.sample(list(class_elements), qtde_pair)

                for k in range(qtde_pair):
                    pairImages.append([images[current_class_elements[k]], images[other_class_elements[k]]])
                    pairLabels.append([0])

        return np.array(pairImages), np.array(pairLabels)

    def load(self):

        print("Loading images and labels from dataset...")
        x_train, y_train, x_val, y_val = self.__get_infos()

        print("Building images pairs...")
        pairTrain, labelTrain = self.__make_pairs(x_train, y_train)
        pairVal, labelVal = self.__make_pairs(x_val, y_val)

        pairTrain, labelTrain = utils.shuffle(pairTrain, labelTrain)

        return pairTrain, labelTrain, pairVal, labelVal





if __name__ == '__main__':
    loader = LoadFics(
        train_size=5,
        dataset_path="/home/agostinho/PyCharm/Siamese/data_fics"
    )

    pairTrain, labelTrain, pairVal, labelVal = loader.load()

    print(pairTrain.shape)
    print(labelTrain.shape)
    print(pairVal.shape)
    print(labelVal.shape)
    # print(len(pairTrain))
    # print(len(labelTrain))
    # print(len(pairVal))
    # print(len(labelVal))
