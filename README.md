# CC_Seal_detection
Mammalian detection model in complex cases + counting

Файл inference_and_model_cfg содержит в себе все необходимое для запуска инференса (Кроме весов модели). Также можно запустить обучение, но для этого необходимо скачать датасет в формате **COCO**.
В самом коде указаны места, где и что нужно менять, чтобы запустить инференс/обучение модели (Например, YOUR_DATASET_TRAIN). Везде нужно указывать пути до файла/папки.

Для обучения модели была использована библиотека detectron2.

Для инференса был использован gradio.

В данный момент, можно менять аннотации в самом gradio после того, как модель обработает изображение (то есть, можно удалить неправильные аннотации и/или разметить тех нерп, которых модель не увидела).

# Запуск приложения
тестировалось на `python=3.10`

-клонировать репозиторий 
`git clone https://github.com/in-black-eye/CC_Seal_detection`
- установить pytorch https://pytorch.org/get-started/locally/
 `cd CC_Seal_detection`
- установить зависимости
- 
  `pip install -r requirements.txt`;
- скачать detectron2
- 
  `git clone https://github.com/facebookresearch/detectron2.git`
-
  `python -m pip install -e detectron2`
 
- скачать pytesseract
- 
  `sudo apt install tesseract-ocr`
  
- скачать веса модели
  `https://drive.google.com/file/d/1Tql1g69Puz4GF7LXNINdiXiy3n7wSkDf/view?usp=sharing`
- указать путь до весов модели в 46 строке interface_and_model_cfg.py
- запустить
  
  `python interface_and_model_cfg.py`
