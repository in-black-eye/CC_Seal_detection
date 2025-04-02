import ray
from ray import serve
from PIL import Image
import numpy as np
import base64
import io
import easyocr
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import nextcloud_client
import supervision as sv
import os
import cv2

ray.init()
serve.start(detached=True, http_options={"host": "0.0.0.0"})

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class MaskRCNNService:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.45

        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
        cfg.MODEL.RETINANET.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

        cfg.INPUT.MIN_SIZE_TEST = 0

        '''Инференс модели'''

        cfg.TEST.PRECISE_BN = True
        cfg.MODEL.WEIGHTS = "model_weights.pth"

        cfg.MODEL.RPN.IOU_THRESHOLDS = [0.1, 0.1]
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(cfg)

    def preprocess(self, image_bytes, img_name):
        img = Image.open(io.BytesIO(base64.b64decode(image_bytes))).convert("RGB")
        img_np = np.array(img)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img.save(img_name)
        public_link = 'https://nextcloud.ds-hub.ru/s/snDJMF835JQ2yoj?path=%2Ftest1'
        nc = nextcloud_client.Client.from_public_link(public_link, folder_password='cXXQ5PXwDg')
        nc.drop_file(img_name)
        os.remove(img_name)
        return img_bgr, img

    async def __call__(self, request):
        data = await request.json()
        image_b64 = data["image"]
        img_name = data["name"]

        try:
            img, img1 = self.preprocess(image_b64, img_name)
        except Exception as e:
            return {"error": f"Image preprocessing failed: {str(e)}"}

        try:
            # Обработка Mask R-CNN
            outputs = self.predictor(img)
            detections = sv.Detections.from_detectron2(outputs)

            # Обработка OCR с очисткой текста
            reader = easyocr.Reader(['en'])
            ocr_results = reader.readtext(img)

            # Фильтрация не-UTF8 символов
            cleaned_ocr = []
            for res in ocr_results:
                try:
                    # Декодируем и кодируем обратно для очистки
                    text = res[1].encode('utf-8', 'ignore').decode('utf-8')
                    cleaned_ocr.append({
                        "text": text,
                        "confidence": float(res[2]),
                        "coordinates": [tuple(map(float, coord)) for coord in res[0]]
                    })
                except Exception as e:
                    print(f"OCR error: {str(e)}")
                    continue

            # Конвертация numpy типов в стандартные Python типы
            response = {
                "boxes": detections.xyxy.tolist(),
                "labels": [int(c) for c in detections.class_id],
                "scores": [float(s) for s in detections.confidence],
                "ocr": cleaned_ocr
            }

            return response

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}

app = MaskRCNNService.bind()
