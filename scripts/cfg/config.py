import os

IMG_SIZE = 256

# BASE_PATH = "/kaggle/input/dataset/new_ds"
BASE_PATH="/Users/polinakuranova/.cache/kagglehub/datasets/radmilasegen/dataset/versions/1/new_ds"
CLASSES_PATH = os.path.join(BASE_PATH, "merged_classes.txt")

PATH_FOR_ASS_DEPTH="/Users/polinakuranova/.cache/kagglehub/datasets/radmilasegen/dataset/versions/1/new_ds/1/depth_assesment/"

MODEL_PATH = "/Users/polinakuranova/uni/ML_in_robotics/multimodal_object_detection/models/10_epochs/rgbd_fasterrcnn_crossmodal_8x8.pth"

MODEL = "base_model" # "base_model | cross_attention_model"

USE_MIDAS = True

EPOCHS = 1
