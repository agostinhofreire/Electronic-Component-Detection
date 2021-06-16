from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import sys


class SiameseModel:
    LOSS_TYPE = ['binary_crossentropy', 'contrastive']
    BACKBONE_TYPE = ['vgg', 'resnet']
    def __init__(self, input_shape, network="vgg", loss="binary_crossentropy"):

        self.input_shape = input_shape
        self.network = network
        self.model = None

        self.loss = loss

    def get_model(self):

        if self.network == "vgg":
            inputs = VGG16(weights='imagenet', input_shape=self.input_shape, include_top=False)
        elif self.network == "resnet":
            inputs = ResNet50(weights='imagenet', input_shape=self.input_shape, include_top=False)
        else:
            print("Invalid model, please choose a valid one -> [vgg, resnet]")
            sys.exit()

        for layer in inputs.layers:
            layer.trainable = False

        flatten = Flatten()(inputs.output)
        dense1 = Dense(512, activation="relu")(flatten)
        dense1 = BatchNormalization()(dense1)
        dense2 = Dense(256, activation="relu")(dense1)
        dense2 = BatchNormalization()(dense2)
        output = Dense(256)(dense2)

        model = Model(inputs.input, output)

        return model


    def euclidean_distance(self, vectors):

        (featsA, featsB) = vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)

        return K.sqrt(K.maximum(sumSquared, K.epsilon()))

    def contrastive_loss(self, y, preds, margin=1):

        y = tf.cast(y, preds.dtype)
        squaredPreds = K.square(preds)
        squaredMargin = K.square(K.maximum(margin - preds, 0))
        loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

        return loss

    def build(self):

        imgA = Input(shape=self.input_shape)
        imgB = Input(shape=self.input_shape)

        featureExtractor = self.get_model()

        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)

        distance = Lambda(self.euclidean_distance)([featsA, featsB])

        if self.loss == "binary_crossentropy":
            outputs = Dense(1, activation="sigmoid")(distance)
            self.model = Model(inputs=[imgA, imgB], outputs=outputs)
            self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        elif self.loss == "contrastive":
            self.model = Model(inputs=[imgA, imgB], outputs=distance)
            self.model.compile(loss=self.contrastive_loss, optimizer="adam", metrics=["accuracy"])

    def train(self, dataset, batch_size=16, epochs=50, path_save="./best_weights.h5"):

        if self.model == None:
            print("Please, build your model first...")
            sys.exit()

        pairTrain, labelTrain, pairVal, labelVal = dataset


        checkpointer = ModelCheckpoint(filepath=path_save,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

        history = self.model.fit(
            [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
            validation_data=([pairVal[:, 0], pairVal[:, 1]], labelVal[:]),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[checkpointer]
        )

        return history



if __name__ == '__main__':
    siamese = SiameseModel((224, 224, 3), "vgg")
    model = siamese.build()
    print(model.summary())