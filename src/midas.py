import torch

class MiDaSDepthEstimator:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Using device: {self.device}")

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

    def predict_from_array(self, rgb):
        input_tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(rgb.shape[0], rgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy()
