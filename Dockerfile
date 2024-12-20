
FROM python:3.10-slim


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-opencv
WORKDIR /app


COPY requirements.txt .

# Зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем PyTorch 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Клонируем репозиторий detectron2 и устанавливаем его
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2 && \
    pip install -e /detectron2

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# весь проект в контейнер
COPY . .

# путь до весов модели 
ENV MODEL_WEIGHTS_PATH=/app/model_weights.pth


CMD ["python", "interface_and_model_cfg.py"]


