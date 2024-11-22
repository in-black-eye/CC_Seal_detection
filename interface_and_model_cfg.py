from PIL import Image

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

setup_logger()
import supervision as sv

import cv2
import json
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog

register_coco_instances("NAME_DATASET_TRAIN", {}, "ANNOTATIONS_FILE",
                        "IMG_ROOT")
register_coco_instances("NAME_DATASET_VALID", {}, "ANNOTATIONS_FILE",
                        "IMG_ROOT")

my_dataset_train_metadata = MetadataCatalog.get("NAME_DATASET_TRAIN")
dataset_dicts = DatasetCatalog.get("NAME_DATASET_TRAIN")

from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("NAME_DATASET_TRAIN",)
cfg.DATASETS.TEST = ("NAME_DATASET_VALID",)
# cfg.INPUT.FORMAT = "RGB"
cfg.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 1e-20
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
# cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 40000
cfg.SOLVER.STEPS = ()
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.CHECKPOINT_PERIOD = 300
cfg.SOLVER.WEIGHT_DECAY = 1e-5

cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.45

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.4
cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.4

cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 0.1  # 2
cfg.MODEL.RPN.LOSS_WEIGHT = 5.0  #
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0  #
cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 3.5
cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.3
cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.8

cfg.OUTPUT_DIR = "YOUR_OUTPUT_DIR"

cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
cfg.MODEL.RETINANET.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True

cfg.INPUT.MIN_SIZE_TRAIN = (480,)
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.45
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
cfg.MODEL.RETINANET.PRIOR_PROB = 0.05
#
cfg.TEST.EVAL_PERIOD = 300

'''Обучение модели'''

# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.train()


'''Инференс модели'''

cfg.TEST.PRECISE_BN = True
cfg.MODEL.WEIGHTS = "YOUR_MODEL_WEIGHTS"
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # Уверенность модели
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.95  # Уверенность модели
# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.1, 0.1]
# cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1]
# # #
predictor = DefaultPredictor(cfg)

import gradio as gr
from gradio_image_annotation import image_annotator
import cv2

sample_annotation = {
    "image": "",
    "boxes": []
}

ann_raw = [[], []]

seals_ann = {1: "seal_rock",
             2: "seal_water"}

seals_colors = {1: (255, 0, 0),
                2: (0, 255, 0)}


# Функция обработки изображения


def process_image(image):
    outputs = predictor(image)
    detections = sv.Detections.from_detectron2(outputs)
    ann_raw[0] = detections.xyxy
    ann_raw[1] = detections.class_id
    v = Visualizer(image[:, :, ::-1],
                   scale=1
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("img/image.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    sample_annotation["image"] = "img/image.jpg"
    return out.get_image()[:, :, ::-1]


# Функция обработки аннотаций
def process_annotations(annotations):
    json_ann = {"boxes": []}

    for i in range(len(annotations[0])):
        coords = annotations[0][i]
        label = seals_ann[annotations[1][i]]
        color = seals_colors[annotations[1][i]]
        json_ann["boxes"].append(
            {
                "xmin": int(coords[0]),
                "ymin": int(coords[1]),
                "xmax": int(coords[2]),
                "ymax": int(coords[3]),
                "label": label,
                "color": color
            })

    return json_ann


def update_ann():
    sample_annotation["image"] = "img/image.jpg"
    dict_with_boxes = process_annotations(ann_raw)
    sample_annotation["boxes"] = dict_with_boxes["boxes"]
    return sample_annotation


with gr.Blocks() as main:
    with gr.Tab("Process"):
        main_process = gr.Interface(
            fn=process_image,  # Функция обработки
            inputs=gr.Image(type="numpy"),
            outputs=gr.Image(type="numpy"),
            title="Обработка изображений",  # Название
            description="Загрузите изображение для обработки.",  # Описание
        )

    with gr.Tab("Object annotation", id="tab_object_annotation"):
        annotator = image_annotator(
            label_list=["seal_water", "seal_rock"],
            label_colors=[(0, 0, 255), (0, 255, 255)],
            show_label=False,
            boxes_alpha=0.2,
            box_thickness=1,
            handle_size=4,
            box_min_size=4
        )
        button_get_img = gr.Button("get_annot")
        button_get_img.click(update_ann, [], annotator)

main.launch()
