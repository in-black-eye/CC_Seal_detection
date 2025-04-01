import requests
import base64
import json
from pathlib import Path


def send_request(image_path):
    # Кодирование изображения в base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Отправка запроса
    response = requests.post(
        "http://localhost:8000/detect",
        json={"image": image_b64}
    )

    return response.json()


def save_json(data, filename="result.json"):
    # Создаем директорию, если её нет
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Записываем данные
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  # indent для читаемого форматирования



# Пример использования
if __name__ == "__main__":
    result = send_request("text-test-result.jpg")
    save_json(result, "detections3.json")
    print(json.dumps(result))
