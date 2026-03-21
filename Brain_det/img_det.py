"""Dataset: 
detection : https://www.kaggle.com/datasets/alirehman8008/brain-tumor-annotated-dataset
classificatin : https://www.kaggle.com/datasets/shreyag1103/brain-mri-scans-for-brain-tumor-classification """


import pickle
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# PATHS
# -----------------------------
test_file = "D://Explore//jupyter//datasets//Brain//class//glioma//Te-gl_0036.jpg"

# -----------------------------
# LOAD MODELS
# -----------------------------
clf_model = load_model("D://Explore//jupyter//img_det//Brain_cls//model.h5")
det_model = YOLO("D://Explore//jupyter//img_det//Brain_det//best.pt")

with open("D://Explore//jupyter//img_det//Brain_cls//label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

index_to_class = ['glioma', 'meningioma', 'notumor','pituitary']
IMG_SIZE = (224, 224)

# -----------------------------
# PIPELINE
# -----------------------------
def classify_and_detect(img_path):
    # ---- Classification ----
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    preds = clf_model.predict(img_array_exp)
    pred_index = np.argmax(preds)
    confidence = np.max(preds)
    pred_class = index_to_class[pred_index]

    # ---- Show classification result ----
    # plt.imshow(img)
    # plt.axis("off")
    # plt.title(f"Class: {pred_class} ({confidence:.2f})")
    # plt.show()

    print(f"Predicted Class : {pred_class}")
    print(f"Confidence      : {confidence:.4f}")

    # ---- If tumor detected → run YOLO ----
    if pred_class != "notumor":
        results = det_model.predict(
            source=img_path,
            conf=0.25,
            device="cpu"
        )

        for r in results:
            annotated = r.plot()  # get plotted image
            plt.imshow(annotated)
            plt.axis("off")
            plt.title(f"YOLO Detection ({pred_class})")
            plt.show()

# -----------------------------
# RUN
# -----------------------------
classify_and_detect(test_file)

