import sys

import numpy as np
import supervision as sv
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QProgressBar, QLabel
from anylabeling import app
from anylabeling.views.mainwindow import MainWindow

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
        MainWindow.resize(1000,800)

        self.any_labeling_widget = AnyLabelingWidget(MainWindow)

        MainWindow.setCentralWidget(self.any_labeling_widget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Seal Detection"))


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


class ProgressPopup(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Загрузка")
        self.setFixedSize(300, 150)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel("Загрузка...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.ok_button = QPushButton("ОК")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.close)
        layout.addWidget(self.ok_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        self.timer.start(50)  # Обновление каждые 50 мс

    def update_progress(self):
        if self.progress_value < 100:
            self.progress_value += 2
            self.progress_bar.setValue(self.progress_value)
        else:
            self.timer.stop()
            self.label.setText("Готово!")
            self.ok_button.setEnabled(True)


class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Устанавливаем UI

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

