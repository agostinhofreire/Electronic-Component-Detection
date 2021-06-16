from src.load_fics_dataset import LoadFics
from src.siamese_model import SiameseModel
from src.utils import plot_training
from src.evaluate_model import evaluate_model

for network in ["vgg", "resnet"]:
    for loss in ["binary_crossentropy", "contrastive"]:
        for train_size in [5, 10, 20]:


            file_name = f"{network}_{loss}_{train_size}"

            print("Starting", file_name)

            loader = LoadFics(
                    train_size=train_size,
                    dataset_path="/home/agostinho/PyCharm/Siamese/data_fics"
                )

            dataset = loader.load() #pairTrain, labelTrain, pairVal, labelVal

            siamese = SiameseModel(
                input_shape=(224, 224, 3),
                network=network,
                loss=loss
            )
            siamese.build()

            history = siamese.train(
                dataset=dataset,
                batch_size=16,
                epochs=50,
                path_save=f"./results/{file_name}.h5",
                path_log=f"./results/{file_name}.log"
            )

            plot_training(history, f"./results/{file_name}.png")

            # siamese.model.load_weights(f"./results/{file_name}.h5")
            #
            # x_train, y_train = loader.get_train_dataset()
            # x_test_path, y_test = loader.get_test_dataset()
            # cm, class_metrics = evaluate_model(siamese.model, x_train, y_train, x_test_path, y_test, f'./results/{file_name}.txt')