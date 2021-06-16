from src.load_fics_dataset import LoadFics
from src.siamese_model import SiameseModel
from src.utils import plot_training

loader = LoadFics(
        train_size=5,
        dataset_path="/home/agostinho/PyCharm/Siamese/data_fics"
    )

dataset = loader.load() #pairTrain, labelTrain, pairVal, labelVal

siamese = SiameseModel((224, 224, 3), "resnet")
siamese.build()

history = siamese.train(
    dataset=dataset,
    batch_size=16,
    epochs=50,
    path_save="./vgg_best_model.h5"
)

plot_training(history)