# CC_Seal_detection
Mammalian detection model in complex cases + counting

Файл inference_and_model_cfg содержит в себе все необходимое для запуска инференса (Кроме весов модели, веса модели(клик). 
Также можно запустить обучение на датасете в формате **COCO**.
В самом коде указаны места, где и что нужно менять, чтобы запустить инференс/обучение модели (Например, YOUR_DATASET_TRAIN).

Для обучения модели была использована библиотека detectron2.

Для веб-интерфейса был использован gradio.

В данный момент, можно менять аннотации в самом gradio после того, как модель обработает изображение (то есть, можно удалить неправильные аннотации и/или разметить тех нерп, которых модель не увидела).

# Запуск приложения на linux-deb(ubuntu) или wsl. 
# cpu only(gpu version soon)
1. `apt-install docker.io`
2. `docker pull arturaz0/seals_detection`
3.`docker run -d -p 7860:7860 arturaz0/seals_detection:v1`
4. В браузере `http://localhost:7860/`
