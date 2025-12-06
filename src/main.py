import torch

from cfg.config import IMG_SIZE, BASE_PATH, CLASSES_PATH
from dataset_install import load_classes, collect_samples
from dataset_process_functions import compute_depth_min_max, collate_fn, train_val_split
from Faster_RCNN_model_with_cross_model_attention import Faster_RCNN_model_with_cross_model_attention
from train import train
from test import show_n_predictions

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    classes = load_classes(CLASSES_PATH)
    print("Classes:", classes)

    samples = collect_samples(classes, BASE_PATH)
    print("Number of valid samples:", len(samples))

    depth_min, depth_max = compute_depth_min_max(samples)
    print("depth_min, depth_max:", depth_min, depth_max)

    train_loader, val_loader = train_val_split(samples, depth_min, depth_max, IMG_SIZE, device, batch_size = 4, collate_fn = collate_fn)

    model = Faster_RCNN_model_with_cross_model_attention(device, classes)

    train(model, train_loader, device, epochs = 10)

    show_n_predictions(model, samples, classes, depth_min, depth_max, IMG_SIZE, device, n = 5, score_thresh=0.6)


