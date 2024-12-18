# CC_Seal_detection
Mammalian detection model in complex cases + counting

Файл inference_and_model_cfg содержит в себе все необходимое для запуска инференса (Кроме весов модели). Также можно запустить обучение, но для этого необходимо скачать датасет в формате **COCO**.
В самом коде указаны места, где и что нужно менять, чтобы запустить инференс/обучение модели (Например, YOUR_DATASET_TRAIN). Везде нужно указывать пути до файла/папки.

Для обучения модели была использована библиотека detectron2.

Для инференса был использован gradio.

В данный момент, можно менять аннотации в самом gradio после того, как модель обработает изображение (то есть, можно удалить неправильные аннотации и/или разметить тех нерп, которых модель не увидела).

# Запуск модели
Для запуска файла необходимо в окружении:
- установить torch, torchvision, torchaudio (`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118`)
- установить зависимости из requirements.txt;
- скачать detectron2 (`git clone https://github.com/facebookresearch/detectron2.git`
                      `python -m pip install -e detectron2`);
- скачать pytesseract (`sudo apt install tesseract-ocr`);
- указать путь до весов модели.
