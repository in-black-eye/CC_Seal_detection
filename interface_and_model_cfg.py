import gradio as gr
import pytesseract
import csv
import cv2
import numpy as np
import supervision as sv
import os
from pathlib import Path
from shutil import rmtree

from zipfile import ZipFile
from gradio_image_annotation import image_annotator

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import time

IMAGE_FOLDER_NAME = 'temp'  # Папка, где хранятся фотографии для обработки
PATH_TO_CSV_TABLE = 'example_table.csv'  # Таблица, в которую записываются данные о фото и количестве нерп.

pytesseract.pytesseract.tesseract_cmd = "PATH_TO_TESSERACT_EXE"

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
cfg.MODEL.WEIGHTS = "MODEL_WEIGHTS"

cfg.MODEL.RPN.IOU_THRESHOLDS = [0.1, 0.1]
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1]

sample_annotation = {
    "image": "",
    "boxes": []
}

annotations = []

seals_ann = {1: "seal_rock",
             2: "seal_water"}

seals_colors = {1: (255, 0, 0),
                2: (0, 255, 0)}

# Переменные для хранения изображений и их порядка
images = []
img_num_now = 0


def process_archive(files, slider_value, progress=gr.Progress()):
    annotations.clear()

    with open(PATH_TO_CSV_TABLE, 'w', newline='', encoding='utf-8') as csvfile:
        nerpwrite = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        nerpwrite.writerow(
            ["Номер фотоловушки (если не переименуем фотографии)", "название фото (номер фото)",
             "дата (формат дд/мм/гггг)",
             "время (формат чч/мм/сс)", "температура с фотоловушки", "Количество нерп на суше",
             "Количество нерп в воде"])
    csvfile.close()

    if not os.path.exists(IMAGE_FOLDER_NAME):
        os.makedirs(IMAGE_FOLDER_NAME)
    else:
        print(f"Папка `{IMAGE_FOLDER_NAME}` уже существует!")

    for path in Path(IMAGE_FOLDER_NAME).glob('*'):
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = slider_value
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = slider_value
    predictor = DefaultPredictor(cfg)

    with ZipFile(files[0], "r") as archive:
        archive.extractall(IMAGE_FOLDER_NAME)

    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    images.clear()
    files_in_temp = os.listdir(IMAGE_FOLDER_NAME)
    for img in progress.tqdm(files_in_temp, desc="Process"):
        time.sleep(0.25)
        images.append(f"{IMAGE_FOLDER_NAME}/" + img)
        image = cv2.imread(f"{IMAGE_FOLDER_NAME}/{img}")
        outputs = predictor(image)
        detections = sv.Detections.from_detectron2(outputs)

        boxes = detections.xyxy
        classes = detections.class_id
        masks = detections.mask
        scores = detections.confidence
        new_boxes = []
        new_classes = []
        new_masks = []
        new_scores = []

        for i in range(len(boxes)):
            length = abs(boxes[i][0] - boxes[i][2])
            width = abs(boxes[i][1] - boxes[i][3])
            area = length * width
            # Производим отбор боксов по их площади
            if area > 10:
                new_boxes.append(boxes[i])
                new_classes.append(classes[i])
                new_masks.append(masks[i])
                new_scores.append(scores[i])

        # Преобразуем списки в массив numpy
        new_boxes = np.array(new_boxes)
        new_masks = np.array(new_masks)
        new_classes = np.array(new_classes)
        new_scores = np.array(new_scores)

        # Преобразуем массивы в sv.Detections
        processed_detections = sv.Detections(xyxy=new_boxes,
                                             mask=new_masks,
                                             class_id=new_classes,
                                             confidence=new_scores)

        boxes_annotations = process_annotations((processed_detections.xyxy, processed_detections.class_id))
        annotations.append(boxes_annotations)
    return "Обработка завершена"


# Функция обработки аннотаций
def process_annotations(annotations):
    json_ann = []

    for i in range(len(annotations[0])):
        coords = annotations[0][i]
        label = seals_ann[annotations[1][i]]
        color = seals_colors[annotations[1][i]]
        json_ann.append(
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
    global img_num_now
    img_num_now = 0
    sample_annotation["image"] = f"{IMAGE_FOLDER_NAME}/" + os.listdir(IMAGE_FOLDER_NAME)[img_num_now]
    sample_annotation["boxes"] = annotations[img_num_now]
    return sample_annotation


def next_img():
    global img_num_now
    if img_num_now + 1 < len(os.listdir(IMAGE_FOLDER_NAME)):
        img_num_now += 1
        sample_annotation["image"] = f"{IMAGE_FOLDER_NAME}/" + os.listdir(IMAGE_FOLDER_NAME)[img_num_now]
        sample_annotation["boxes"] = annotations[img_num_now]
    return sample_annotation


def prev_img():
    global img_num_now

    if img_num_now - 1 >= 0:
        img_num_now -= 1
        sample_annotation["image"] = f"{IMAGE_FOLDER_NAME}/" + os.listdir(IMAGE_FOLDER_NAME)[img_num_now]
        sample_annotation["boxes"] = annotations[img_num_now]
    return sample_annotation


def save_annot(annotator):
    global img_num_now
    annotations[img_num_now] = annotator["boxes"]


def get_results():
    global img_num_now
    img_cv = cv2.imread(f"{IMAGE_FOLDER_NAME}/{os.listdir(IMAGE_FOLDER_NAME)[img_num_now]}")
    img_cv = cv2.resize(img_cv, (0, 0), fx=0.5, fy=0.5)
    height, width, _ = img_cv.shape
    cropped_img_cv = img_cv[height - 50:height]
    count_seal_rock = 0
    count_seal_water = 0
    for box in annotations[img_num_now]:
        if box["label"] == "seal_rock":
            count_seal_rock += 1
        else:
            count_seal_water += 1

    img_rgb = cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_string(img_rgb, config='--psm 6').split('\n')[-2].split()

    with open(PATH_TO_CSV_TABLE, 'a', newline='', encoding='utf-8') as csvfile:
        nerpwrite = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        nerpwrite.writerow(
            [0, os.listdir(IMAGE_FOLDER_NAME)[img_num_now], data[3], data[4], data[-2], count_seal_rock,
             count_seal_water])
    csvfile.close()


with gr.Blocks(theme=gr.themes.Soft()) as main:
    with gr.Tab("Process"):
        confidence_slider = gr.Slider(0, 1, value=0.9, step=0.05, label="Уверенность модели")
        zip_archive_input = gr.File(file_count="multiple")
        submit_btn = gr.Button("Submit", variant="primary")
        status = gr.Text(show_label=False)
        submit_btn.click(process_archive, [zip_archive_input, confidence_slider], status)

    with gr.Tab("Object annotation", id="tab_object_annotation"):
        button_get_img = gr.Button("Показать аннотации", variant='primary')
        with gr.Row():
            prev_btn = gr.Button("Предыдущая фотография", variant='huggingface')
            next_btn = gr.Button("Следующая фотография", variant='huggingface')
        with gr.Row():
            save_annot_btn = gr.Button('Сохранить информацию в таблицу', variant='huggingface')
            download_btn = gr.DownloadButton("Скачать таблицу", visible=True,
                                             value=PATH_TO_CSV_TABLE, variant='secondary')

        annotator = image_annotator(
            label_list=["seal_water", "seal_rock"],
            label_colors=[(255, 0, 0), (0, 255, 0)],
            show_label=False,
            boxes_alpha=0.1,
            box_thickness=1,
            handle_size=4,
            box_min_size=4
        )

        prev_btn.click(prev_img, [], annotator)
        next_btn.click(next_img, [], annotator)
        button_get_img.click(update_ann, [], annotator)
        save_annot_btn.click(get_results, [], [])
        annotator.change(save_annot, annotator)

main.launch()
