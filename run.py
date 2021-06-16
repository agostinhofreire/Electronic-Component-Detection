from src.load_fics_dataset import LoadFics
from src.siamese_model import SiameseModel
from src.utils import plot_training
from src.evaluate_model import evaluate_model

loader = LoadFics(
        train_size=5,
        #dataset_path="/home/agostinho/PyCharm/Siamese/data_fics"
        dataset_path = "D:\\PCBs Datasets\\FICS PCB\\FICS_CROPS_V3"
    )

dataset = loader.load() #pairTrain, labelTrain, pairVal, labelVal

siamese = SiameseModel(
    input_shape=(224, 224, 3),
    network="vgg",
    loss="binary_crossentropy"
)
siamese.build()

history = siamese.train(
    dataset=dataset,
    batch_size=16,
    epochs=50,
    path_save="./vgg_best_model.h5"
)

plot_training(history)

x_train, y_train = loader.get_train_dataset()
x_test_path, y_test = loader.get_test_dataset()
cm, class_metrics = evaluate_model(siamese, x_train, y_train, x_test_path, y_test, './vgg_binary.txt')