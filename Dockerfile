FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-opencv
WORKDIR /app

COPY requirements.txt .

# Зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Клонируем репозиторий detectron2 и устанавливаем его
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2 && \
    pip install -e /detectron2

EXPOSE 8000

# весь проект в контейнер
COPY . .

# путь до весов модели 
ENV MODEL_WEIGHTS_PATH=/app/model_weights.pth

CMD ["ray", "start", "--head", "--num-gpus=1"]
CMD ["serve", "run", "seal_server:app", "--name=mask_rcnn_cpu", "--route-prefix=/detect"]


