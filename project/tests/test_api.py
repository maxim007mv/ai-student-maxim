"""
Тесты для FastAPI-сервиса.

Покрывают:
- Health-check
- Predict (с моделью и без)
- Валидация входных данных
- Ошибки (400, 422, 503)
- Prometheus-метрики
- Response schema
"""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)


# Health-check

def test_health_check():
    """Проверка доступности health-check эндпоинта."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data


def test_health_response_schema():
    """Проверка структуры ответа health-check."""
    response = client.get("/health")
    data = response.json()
    assert isinstance(data["status"], str)
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["device"], str)


# Predict — валидация

def test_predict_no_file():
    """POST /predict без файла — 422 Unprocessable Entity."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_wrong_content_type():
    """POST /predict с не-изображением — 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    # Модель может быть не загружена → 503, либо не изображение → 400
    assert response.status_code in (400, 503)


def test_predict_empty_file():
    """POST /predict с пустым файлом."""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code in (400, 422, 500, 503)


# Predict — с реальным изображением (если модель загружена)

@pytest.fixture
def sample_image() -> bytes:
    """Генерирует тестовое RGB-изображение 224x224."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.mark.skipif(
    not Path("outputs/best_model.pth").exists(),
    reason="Модель не обучена — чекпоинт отсутствует",
)
def test_predict_with_image(sample_image):
    """Инференс на случайном изображении (требуется модель)."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")},
    )
    if response.status_code == 200:
        data = response.json()
        assert "filename" in data
        assert "detections" in data
        assert "count" in data
        assert isinstance(data["detections"], list)
        assert data["count"] == len(data["detections"])
    elif response.status_code == 503:
        pytest.skip("Модель не загружена в сервисе")
    else:
        pytest.fail(f"Неожиданный статус-код: {response.status_code}")


# Prometheus-метрики

def test_metrics_endpoint():
    """Проверка эндпоинта /metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "api_requests_total" in content
    assert "api_request_duration_seconds" in content
    assert "api_predict_count" in content
    assert "api_model_loaded" in content


# Smoke-тест: несколько запросов подряд

def test_smoke_multiple_health_checks():
    """Smoke-тест: последовательные health-check запросы."""
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200


def test_smoke_predict_sequence(sample_image):
    """Smoke-тест: последовательные predict-запросы."""
    statuses = []
    for _ in range(3):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")},
        )
        statuses.append(response.status_code)
    # Все запросы должны вернуть одинаковый статус
    assert len(set(statuses)) == 1, f"Статусы различаются: {statuses}"


# Граничные случаи

def test_predict_corrupted_jpeg():
    """POST /predict с битым JPEG."""
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100), "image/jpeg")},
    )
    assert response.status_code in (400, 422, 500, 503)


def test_predict_large_filename():
    """POST /predict с длинным именем файла."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    long_name = "a" * 256 + ".jpg"
    response = client.post(
        "/predict",
        files={"file": (long_name, buf, "image/jpeg")},
    )
    assert response.status_code in (200, 503)
