import gradio as gr
import pytesseract
import csv
import cv2
import numpy as np
import supervision as sv
import os
from pathlib import Path
from shutil import rmtree, make_archive

from zipfile import ZipFile
from gradio_image_annotation import image_annotator

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

IMAGE_FOLDER_NAME = 'temp'  # Папка, где хранятся фотографии для обработки
PATH_TO_EXCEL_TABLE = 'seal_data.xlsx'  # Таблица, в которую записываются данные о фото и количестве нерп.
PATH_TO_SAVE_ANNOTATIONS = 'saved_annotations'  # Папка, где будут храниться аннотации к фото.
PATH_TO_ZIP_ARCHIVE = "annotated_images"  # Название архива с аннотациями
DOWNLOAD_PATH_TO_ZIP_ARCHIVE = "annotated_images.zip"  # Путь к архиву с аннотациями

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

columns = ["Номер фотоловушки", "название фото (номер фото)",
           "дата (формат дд/мм/гггг)",
           "время (формат чч/мм/сс)", "температура с фотоловушки", "Количество нерп на суше", "Количество нерп в воде"]

data_rows = []


def zipdir(path):
    make_archive(PATH_TO_ZIP_ARCHIVE, "zip", path)


def process_archive(files, slider_value, progress=gr.Progress()):
    global sample_annotation
    annotations.clear()

    sample_annotation = {
        "image": "",
        "boxes": []
    }

    if not os.path.exists(IMAGE_FOLDER_NAME):
        os.makedirs(IMAGE_FOLDER_NAME)
    else:
        print(f"Папка `{IMAGE_FOLDER_NAME}` уже существует!")

    if not os.path.exists(PATH_TO_SAVE_ANNOTATIONS):
        os.makedirs(PATH_TO_SAVE_ANNOTATIONS)
        os.makedirs(f"{PATH_TO_SAVE_ANNOTATIONS}/image")
    else:
        print(f"Папка `{PATH_TO_SAVE_ANNOTATIONS}` уже существует!")

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
    progress(0.05)
    images.clear()
    files_in_temp = os.listdir(IMAGE_FOLDER_NAME)
    for img in progress.tqdm(files_in_temp, desc="Process"):
        images.append(f"{IMAGE_FOLDER_NAME}/" + img)
        # print(f"{IMAGE_FOLDER_NAME}/{img}")
        image = cv2.imread(f"{IMAGE_FOLDER_NAME}/{img}")
        outputs = predictor(image)
        try:
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
                if area > 10:
                    new_boxes.append(boxes[i])
                    new_classes.append(classes[i])
                    new_masks.append(masks[i])
                    new_scores.append(scores[i])

            if new_boxes:
                new_boxes = np.array(new_boxes)
                new_masks = np.array(new_masks)
                new_classes = np.array(new_classes)
                new_scores = np.array(new_scores)

                processed_detections = sv.Detections(xyxy=new_boxes,
                                                     mask=new_masks,
                                                     class_id=new_classes,
                                                     confidence=new_scores)
                boxes_annotations = process_annotations((processed_detections.xyxy, processed_detections.class_id))
            else:
                boxes_annotations = process_annotations(
                    (np.empty((0, 4)), np.empty(0)))

            annotations.append(boxes_annotations)
        except Exception as e:
            print(f"Ошибка при обработке изображения {img}: {e}")
            annotations.append(process_annotations((np.empty((0, 4)), np.empty(0))))
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


def download_annotations():
    zipdir(PATH_TO_SAVE_ANNOTATIONS)
    return DOWNLOAD_PATH_TO_ZIP_ARCHIVE


def get_results(index):
    img_cv = cv2.imread(f"{IMAGE_FOLDER_NAME}/{os.listdir(IMAGE_FOLDER_NAME)[index]}")
    img_cv = cv2.resize(img_cv, (0, 0), fx=0.5, fy=0.5)
    height, width, _ = img_cv.shape
    cropped_img_cv = img_cv[height - 50:height]
    count_seal_rock = 0
    count_seal_water = 0
    for box in annotations[index]:
        if box["label"] == "seal_rock":
            count_seal_rock += 1
        else:
            count_seal_water += 1

    img_rgb = cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_string(img_rgb, config='--psm 6').split('\n')[-2].split()

    photo = os.listdir(IMAGE_FOLDER_NAME)[index]
    row_data = [photo.split("_")[0], photo.split("_")[1], data[3], data[4], data[-2], count_seal_rock, count_seal_water]
    data_rows.append(row_data)


def get_csv_file():
    classes_id = []
    boxes = []
    for j in range(len(annotations)):
        get_results(j)
        for i in range(len(annotations[j])):
            classes_id.append(1) if annotations[j][i]['label'] == 'seal_rock' else classes_id.append(2)
            boxes.append(np.array([annotations[j][i]['xmin'], annotations[j][i]['ymin'],
                                   annotations[j][i]['xmax'], annotations[j][i]['ymax']]))

        if classes_id:
            classes_id = np.array(classes_id)
            boxes = np.array(boxes)

        else:
            classes_id = np.array(classes_id)
            boxes = np.empty((0, 4))

        detections = sv.Detections(xyxy=boxes, class_id=classes_id)

        detectionDataset = sv.DetectionDataset(classes=['seal_water', 'seal_rock'],
                                               images={f"{images[j]}": cv2.imread(images[j])},
                                               annotations={f"{images[j]}": detections})

        detectionDataset.as_coco(images_directory_path=f'{PATH_TO_SAVE_ANNOTATIONS}/image',
                                 annotations_path=f'{PATH_TO_SAVE_ANNOTATIONS}/{images[j][:-4]}.json')

        with open(f'{PATH_TO_SAVE_ANNOTATIONS}/{images[j][:-4]}.json', 'r') as f:
            data = json.load(f)

        data['categories'] = [{"id": 0, "name": "seal", "supercategory": "none"},
                              {"id": 1, "name": "seal_rocks", "supercategory": "seal"},
                              {"id": 2, "name": "seal_water", "supercategory": "seal"}]

        with open(f'{PATH_TO_SAVE_ANNOTATIONS}/{images[j][:-4]}.json', 'w') as f:
            json.dump(data, f, indent=4)

        classes_id = []
        boxes = []

    info_dataframe = pd.DataFrame(data_rows, columns=columns)
    info_dataframe.to_excel(PATH_TO_EXCEL_TABLE, index=False)

    return PATH_TO_EXCEL_TABLE


with gr.Blocks(theme=gr.themes.Soft(), css_paths='styles.css') as main:
    with gr.Tab("Process"):
        confidence_slider = gr.Slider(0, 1, value=0.9, step=0.05, label="Уверенность модели")
        zip_archive_input = gr.File(file_count="multiple")
        submit_btn = gr.Button("Submit", variant="primary")
        status = gr.Text(show_label=False)
        submit_btn.click(process_archive, [zip_archive_input, confidence_slider], status)

    with gr.Tab("Object annotation", id="tab_object_annotation"):
        with gr.Row():
            button_get_img = gr.Button("Показать аннотации", variant='primary', scale=6)
            button_get_annotations = gr.Button("Скачать аннотации", scale=1)
            button_get_annotations_hidden = gr.DownloadButton(visible=False, elem_id='button_get_annotations_hidden')
        with gr.Row():
            prev_btn = gr.Button("Предыдущая фотография", variant='huggingface')
            next_btn = gr.Button("Следующая фотография", variant='huggingface')
        with gr.Row():
            download_btn = gr.Button("Скачать таблицу")
            download_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_btn_hidden")

        annotator = image_annotator(
            label_list=["seal_water", "seal_rock"],
            label_colors=[(0, 255, 0), (255, 0, 0)],
            show_label=False,
            boxes_alpha=0.1,
            box_thickness=1,
            handle_size=4,
            box_min_size=4
        )

        prev_btn.click(prev_img, [], annotator)
        next_btn.click(next_img, [], annotator)
        button_get_img.click(update_ann, [], annotator)
        button_get_annotations.click(download_annotations, [], [button_get_annotations_hidden]).then(fn=None,
                                                                                                     inputs=None,
                                                                                                     outputs=None,
                                                                                                     js="() => document.querySelector('#button_get_annotations_hidden').click()")
        annotator.change(save_annot, annotator)
        download_btn.click(fn=get_csv_file, inputs=None, outputs=[download_btn_hidden]).then(fn=None, inputs=None,
                                                                                             outputs=None,
                                                                                             js="() => document.querySelector('#download_btn_hidden').click()")

main.launch(root_path='/gradio-demo')
