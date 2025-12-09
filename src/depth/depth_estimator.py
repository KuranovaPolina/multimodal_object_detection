# 🎯 Этот файл превращает обычную RGB-фотографию → в карту глубины.

# Он нужен для:

# сравнения с RealSense глубиной

# оценки качества глубины

# экспериментов с мультимодальным детектором

# построения heatmap ошибок

# анализа сцен с окклюзиями

import torch
import cv2
import numpy as np

class MiDaSDepthEstimator:
    def __init__(self, device=None):
        # Если user НЕ указал device → выбираем автоматически
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Using device: {self.device}")

        # Загружаем MiDaS модель
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        # Трансформации MiDaS
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

    def predict_from_array(self, rgb):
        """
        rgb — numpy array (H, W, 3)
        Возвращает depth карту как numpy float32.
        """

        input_tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Приводим размер предсказания к исходному
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(rgb.shape[0], rgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()
        return depth
