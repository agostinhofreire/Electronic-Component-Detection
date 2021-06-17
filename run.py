from src.load_fics_dataset import LoadFics
from src.siamese_model import SiameseModel
from src.utils import plot_training
from src.evaluate_model import evaluate_model
import os

if not os.path.exists("./results"):
    os.makedirs("./results")

for network in ["vgg"]:
    for loss in ["contrastive", "binary_crossentropy"]:
        for train_size in [5, 10, 15]:

            file_name = f"{network}_{loss}_{train_size}"

            print("Starting", file_name)

            loader = LoadFics(
                    train_size=train_size,
                    #dataset_path="/home/agostinho/PyCharm/Siamese/data_fics"
                    dataset_path="/mnt/hdd/PCBs Datasets/FICS PCB/FICS_CROPS_V3"
                )

            siamese = SiameseModel(
                input_shape=(224, 224, 3),
                network=network,
                loss=loss
            )
            siamese.build()

            dataset = loader.load() #pairTrain, labelTrain, pairVal, labelVal
            history = siamese.train(
                dataset=dataset,
                batch_size=16,
                epochs=50,
                verbose=1,
                path_save=f"./results/{file_name}.h5",
                path_log=f"./results/{file_name}.log"
            )

            plot_training(history, f"./results/{file_name}.png")

            siamese.model.load_weights(f"./results/{file_name}.h5")
            
            x_train, y_train = loader.get_train_dataset()
            x_test_path, y_test = loader.get_test_dataset()
            cm, class_metrics = evaluate_model(siamese.model, x_train, y_train, x_test_path, y_test, f'./results/{file_name}.txt', f'./results/{file_name}.csv')
            del loader
            del siamese