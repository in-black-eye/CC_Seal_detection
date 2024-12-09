from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import os
setup_logger()

register_coco_instances("TRAIN_DATA_NAME", {}, "ANNOTATIONS_PATH.JSON",
                        "IMAGE_TRAIN_PATH")
register_coco_instances("VALID_DATA_NAME", {}, "ANNOTATIONS_PATH.JSON",
                        "IMAGE_VALID_PATH")

my_dataset_train_metadata = MetadataCatalog.get("TRAIN_DATA_NAME")
dataset_dicts = DatasetCatalog.get("VALID_DATA_NAME")


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("TRAIN_DATA_NAME",)
cfg.DATASETS.TEST = ("VALID_DATA_NAME",)
# cfg.INPUT.FORMAT = "RGB"
cfg.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 1e-20
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
# cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 80000
cfg.SOLVER.STEPS = ()
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.CHECKPOINT_PERIOD = 600
cfg.SOLVER.WEIGHT_DECAY = 1e-7

cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.45

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.4
cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.4

cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 0.1
cfg.MODEL.RPN.LOSS_WEIGHT = 5.0
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 3.5
cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.3
cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.8

cfg.OUTPUT_DIR = "OUTPUT_DIR_PATH"

cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
cfg.MODEL.RETINANET.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True

cfg.INPUT.MIN_SIZE_TRAIN = (480,)
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.4
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
cfg.MODEL.RETINANET.PRIOR_PROB = 0.05
cfg.TEST.EVAL_PERIOD = 300
# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.train()
