import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import time

def classify_sample(model, anchors, labels_anchors, sample):
    pairs = [[anchor, sample] for anchor in anchors]
    pairs = np.array(pairs)
    sim = model.predict([pairs[:, 0], pairs[:, 1]])
    sim = [x[0] for x in list(sim)]
    max_similarity_index = sim.index(min(sim))
    return labels_anchors[max_similarity_index], sim

def evaluate_model(model, x_train, y_train, x_test_path, y_test, output_file=None, output_predict=None):
    test_preds = []
    len_test_dataset = len(x_test_path)
    pred_file = None
    if output_predict:
        pred_file = open(output_predict, 'w')

    for i, path in enumerate(x_test_path):
        start = time.time()
        img = cv2.imread(path, 1) / 255.
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        label, sim = classify_sample(model, x_train, y_train, img)
        test_preds.append(label)
        end = time.time()
        if pred_file:
            pred_file.write(str(path) + ',')
            sim = [str(x) for x in sim]
            pred_file.write(','.join(sim))
            pred_file.write('\n')
        print(end - start, '-', path,  i, '/', len_test_dataset, ' - ', label, ':', y_test[i])
    
    if pred_file:
        pred_file.close()
    cm_matrix = confusion_matrix(y_test, test_preds)
    class_metrics = classification_report(y_test, test_preds)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(str(class_metrics))
            f.write('\n')
            f.write(str(cm_matrix))
    return cm_matrix, class_metrics 
