import pytesseract
import csv
import cv2
import numpy as np
import supervision as sv
import os
from pathlib import Path
from shutil import rmtree, make_archive

from zipfile import ZipFile

from PyQt5.QtCore import QThread, pyqtSignal
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import json
import warnings
import math
from anylabeling.views.mainwindow import MainWindow
from anylabeling import app


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QSizePolicy

import sys

IMAGE_FOLDER_NAME = 'temp'  # Папка, где хранятся фотографии для обработки
PATH_TO_CSV_TABLE = 'example_table.csv'  # Таблица, в которую записываются данные о фото и количестве нерп.
PATH_TO_SAVE_ANNOTATIONS = 'saved_annotations'  # Папка, где будут храниться аннотации к фото.
PATH_TO_ZIP_ARCHIVE = "annotated_images"  # Название архива с аннотациями
DOWNLOAD_PATH_TO_ZIP_ARCHIVE = "annotated_images.zip"  # Путь к архиву с аннотациями


class AnyLabelingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Создаем макет для вкладки
        layout = QVBoxLayout(self)

        # Создаем экземпляр AnyLabeling
        self.any_labeling = MainWindow(app)  # Подключаем основной интерфейс AnyLabeling

        # Добавляем AnyLabeling в макет
        layout.addWidget(self.any_labeling)

        self.setLayout(layout)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800,600)

        self.centralwidget = QtWidgets.QTabWidget(MainWindow)
        self.centralwidget.setObjectName("central widget")

        self.tab_process = QtWidgets.QWidget()
        self.tab_process.setObjectName("tab_process")
        self.tab_annot = QtWidgets.QWidget()
        self.tab_annot.setObjectName("tab_annot")

        self.centralwidget.addTab(self.tab_process, "")
        self.centralwidget.addTab(self.tab_annot, "")

        # Используем QVBoxLayout для tab_process
        self.layout = QVBoxLayout(self.tab_process)

        self.slider_confidence = QtWidgets.QSlider(self.tab_process)
        self.slider_confidence.setOrientation(QtCore.Qt.Horizontal)
        self.slider_confidence.setObjectName("slider_confidence")

        self.btn_drop_file = QtWidgets.QPushButton(self.tab_process)
        self.btn_drop_file.setObjectName("btn_drop_file")

        self.btn_submit = QtWidgets.QPushButton(self.tab_process)
        self.btn_submit.setObjectName("btn_submit")

        self.progressBar = QtWidgets.QProgressBar(self.tab_process)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        # Добавляем элементы в layout
        self.layout.addWidget(self.slider_confidence)
        self.layout.addWidget(self.btn_drop_file)
        self.layout.addWidget(self.btn_submit)
        self.layout.addWidget(self.progressBar)

        self.tab_process.setLayout(self.layout)

        # Добавляем AnyLabelingWidget на вкладку "Object Annotation"
        self.any_labeling_widget = AnyLabelingWidget(self.tab_annot)

        self.layout_annot = QVBoxLayout(self.tab_annot)
        self.layout_annot.addWidget(self.any_labeling_widget)

        self.tab_annot.setLayout(self.layout_annot)


        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Seal Detection"))
        self.btn_drop_file.setText(_translate("MainWindow", "Перетащиет файл сюда или Нажмите, чтобы загрузить"))
        self.btn_submit.setText(_translate("MainWindow", "Обработать"))
        self.centralwidget.setTabText(0, _translate("MainWindow", "Process"))
        self.centralwidget.setTabText(1, _translate("MainWindow", "Object Annotation"))


class ImageProcessingThread(QThread):
    progressUpdated = pyqtSignal(int)  # Сигнал для обновления прогресса
    processingFinished = pyqtSignal()  # Сигнал для завершения процесса

    def __init__(self, zip_file_path, slider_value, cfg, annotations, seals_ann, seals_colors):
        super().__init__()
        self.zip_file_path = zip_file_path
        self.slider_value = slider_value
        self.cfg = cfg
        self.annotations = annotations
        self.seals_ann = seals_ann
        self.seals_colors = seals_colors

    def run(self):
        self.process_archive(self.zip_file_path, self.slider_value)

    def stop(self):
        """Метод для остановки потока"""
        self._is_running = False

    def process_archive(self, file_path, slider_value):
        self.annotations.clear()

        with open(PATH_TO_CSV_TABLE, 'w', newline='', encoding='utf-8') as csvfile:
            nerpwrite = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nerpwrite.writerow(
                ["Номер фотоловушки (если не переименуем фотографии)", "название фото (номер фото)",
                 "дата (формат дд/мм/гггг)", "время (формат чч/мм/сс)", "температура с фотоловушки",
                 "Количество нерп на суше", "Количество нерп в воде"])

        if not os.path.exists(IMAGE_FOLDER_NAME):
            os.makedirs(IMAGE_FOLDER_NAME)

        if not os.path.exists(PATH_TO_SAVE_ANNOTATIONS):
            os.makedirs(PATH_TO_SAVE_ANNOTATIONS)
            os.makedirs(f"{PATH_TO_SAVE_ANNOTATIONS}/image")

        for path in Path(IMAGE_FOLDER_NAME).glob('*'):
            if path.is_dir():
                rmtree(path)
            else:
                path.unlink()

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = slider_value
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = slider_value
        predictor = DefaultPredictor(self.cfg)

        with ZipFile(file_path, "r") as archive:
            archive.extractall(IMAGE_FOLDER_NAME)

        files_in_temp = os.listdir(IMAGE_FOLDER_NAME)
        files_count = len(files_in_temp)

        for num, img in enumerate(files_in_temp):
            image = cv2.imread(f"{IMAGE_FOLDER_NAME}/{img}")
            self.predict_seals(predictor, image)

            progress = int((num + 1) * (100 // files_count))
            self.progressUpdated.emit(progress)

        self.processingFinished.emit()  # Завершаем обработку

    def predict_seals(self, predictor, image):
        outputs = predictor(image)
        try:
            detections = sv.Detections.from_detectron2(outputs)
            boxes = detections.xyxy
            classes = detections.class_id
            masks = detections.mask
            scores = detections.confidence

            new_boxes, new_classes, new_masks, new_scores = [], [], [], []
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

                processed_detections = sv.Detections(xyxy=new_boxes, mask=new_masks, class_id=new_classes,
                                                     confidence=new_scores)
                boxes_annotations = self.process_annotations((processed_detections.xyxy, processed_detections.class_id))
            else:
                boxes_annotations = self.process_annotations((np.empty((0, 4)), np.empty(0)))

            self.annotations.append(boxes_annotations)
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            self.annotations.append(self.process_annotations((np.empty((0, 4)), np.empty(0))))

    def process_annotations(self, annotations):
        json_ann = []
        for i in range(len(annotations[0])):
            coords = annotations[0][i]
            label = self.seals_ann[annotations[1][i]]
            color = self.seals_colors[annotations[1][i]]
            json_ann.append({
                "xmin": int(coords[0]),
                "ymin": int(coords[1]),
                "xmax": int(coords[2]),
                "ymax": int(coords[3]),
                "label": label,
                "color": color
            })
        return json_ann


class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Устанавливаем UI

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.INPUT.MASK_FORMAT = "polygon"
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
        self.cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.45

        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
        self.cfg.MODEL.RETINANET.NUM_CLASSES = 3
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

        self.cfg.INPUT.MIN_SIZE_TEST = 0

        '''Инференс модели'''

        self.cfg.TEST.PRECISE_BN = True
        self.cfg.MODEL.WEIGHTS = "model_0077399.pth"
        self.cfg.MODEL.DEVICE = "cpu"


        self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.1, 0.1]
        self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1]

        self.sample_annotation = {
            "image": "",
            "boxes": []
        }

        self.annotations = []

        self.seals_ann = {1: "sr",
                          2: "sw"}

        self.seals_colors = {1: (255, 0, 0),
                             2: (0, 255, 0)}

        # Переменные для хранения изображений и их порядка
        self.images = []
        self.img_num_now = 0

        self.zip_file_path = ""
        self.thread = None

        self.btn_drop_file.clicked.connect(self.open_file_dialog)
        self.btn_submit.clicked.connect(self.submit_images)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Все файлы (*);;Текстовые файлы (*.txt)")

        if file_path:
            self.btn_drop_file.setText(f"Выбран файл: {file_path}")
            self.zip_file_path = file_path

    def submit_images(self):
        if self.zip_file_path:
            # Создаем и запускаем поток для обработки изображений
            if self.thread and self.thread.isRunning():
                pass
            else:
                self.thread = ImageProcessingThread(self.zip_file_path, 0.9, self.cfg, self.annotations, self.seals_ann, self.seals_colors)
                self.thread.progressUpdated.connect(self.update_progress_bar)
                self.thread.processingFinished.connect(self.on_processing_finished)
                self.thread.start()

    def stop_processing(self):
        """Останавливает поток"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()  # Меняем флаг, поток завершает работу
            self.thread.quit()  # Пытаемся корректно завершить поток
            self.thread.wait()  # Ждем завершения потока
            print("Поток остановлен")

    def update_progress_bar(self, progress):
        self.progressBar.setValue(progress)

    def on_processing_finished(self):
        print("Обработка изображений завершена!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

