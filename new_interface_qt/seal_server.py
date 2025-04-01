# server.py (CPU-версия)
from ray import serve
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import base64
import io


# Убираем упоминания GPU из декоратора
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4})  # <-- Удален ray_actor_options
class MaskRCNNService:
    def __init__(self):
        # Явно указываем устройство CPU
        self.device = torch.device("cpu")

        # Загрузка модели для CPU
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=2)

        state_dict = torch.load(
            "model_0077399.pth",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_bytes):
        # Обработка изображения для CPU
        img = Image.open(io.BytesIO(base64.b64decode(image_bytes))).convert("RGB")
        img = F.to_tensor(img)
        return img.unsqueeze(0).to(self.device)  # <-- Заменяем .cuda() на .to(device)

    async def __call__(self, request):
        data = await request.json()
        image_b64 = data["image"]

        input_tensor = self.preprocess(image_b64)

        with torch.no_grad():
            predictions = self.model(input_tensor)[0]

        return {
            "boxes": predictions["boxes"].cpu().numpy().tolist(),
            "labels": predictions["labels"].cpu().numpy().tolist(),
            "scores": predictions["scores"].cpu().numpy().tolist(),
            #"masks": predictions["masks"].cpu().numpy().astype(np.uint8).tolist()
        }


app = MaskRCNNService.bind()