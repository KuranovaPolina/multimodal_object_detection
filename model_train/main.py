import torch

from cfg.config import IMG_SIZE, BASE_PATH, CLASSES_PATH, MODEL, EPOCHS, USE_MIDAS
from src.dataset_download_functions import load_classes, collect_samples
from src.dataset_process_functions import compute_depth_min_max, train_val_split
from src.models import Faster_RCNN_model_with_cross_model_attention, Faster_RCNN_model
from src.train import train
from src.test import show_n_predictions

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    classes = load_classes(CLASSES_PATH)
    print("Classes:", classes)

    samples = collect_samples(classes, BASE_PATH, USE_MIDAS)
    print("Number of valid samples:", len(samples))

    depth_min, depth_max = compute_depth_min_max(samples)
    print("depth_min, depth_max:", depth_min, depth_max)

    train_loader, val_loader = train_val_split(samples, depth_min, depth_max, IMG_SIZE, device, batch_size = 4)

    if MODEL == "cross_attention_model":
        model = Faster_RCNN_model_with_cross_model_attention(device, classes)
    elif MODEL == "base_model":
        model = Faster_RCNN_model(device, classes)

    train(model, train_loader, device, epochs = EPOCHS)

    if MODEL == "cross_attention_model":
        torch.save(model.state_dict(), "rgbd_fasterrcnn_crossmodal_8x8.pth")
        print("Model saved: rgbd_fasterrcnn_crossmodal_8x8.pth")
    elif MODEL == "base_model":
        torch.save(model.state_dict(), "rgbd_fasterrcnn_8x8.pth")
        print("Model saved: rgbd_fasterrcnn_8x8.pth")

    show_n_predictions(model, samples, classes, depth_min, depth_max, IMG_SIZE, device, n = 5, score_thresh=0.6)
