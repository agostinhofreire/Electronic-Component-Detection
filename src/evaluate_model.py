import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import cv2

def classify_sample(model, anchors, labels_anchors, sample):
    similarities = []
    sample = np.expand_dims(sample, axis=0)
    for anchor in anchors:
        sim = model.predict([np.expand_dims(anchor, axis=0), sample])
        similarities.append(sim)
    
    max_similarity_index = similarities.index(max(similarities))
    return labels_anchors[max_similarity_index]

def evaluate_model(model, x_train, y_train, x_test_path, y_test, class_name=[""], output_file=None):
    test_preds = []
    for path in x_test_path:
        img = cv2.imread(path, 1) / 255.
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        label = classify_sample(model, x_train, y_train, img)
        test_preds.append(label)
    
    cm_matrix = confusion_matrix(y_test, test_preds, lables=class_name)
    class_metrics = classification_report(y_test, test_preds, target_names=class_name)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(str(class_metrics))
            f.write('\n')
            f.write(str(cm_matrix))
    return cm_matrix, class_metrics 
