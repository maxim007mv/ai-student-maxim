"""
Car Damage Detection API — FastAPI-сервис с наблюдаемостью.

Эндпоинты:
- GET  /health          — проверка работоспособности
- POST /predict         — сегментация повреждений на изображении
- GET  /metrics         — Prometheus-метрики
"""

from __future__ import annotations

import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from car_damage_segmentation.inference import predict_instances, summarize_predictions
from car_damage_segmentation.modeling import load_model_from_checkpoint
from car_damage_segmentation.utils import resolve_device, setup_logger

# --- Логирование ---
log = setup_logger("api", level=logging.INFO, log_file="outputs/api.log")

# --- FastAPI ---
app = FastAPI(
    title="Car Damage Detection API",
    description="REST API для сегментации повреждений автомобиля",
    version="1.0.0",
)

# --- Prometheus-метрики (без внешних зависимостей) ---
# Реализованы вручную для избежания тяжёлой зависимости prometheus_client
# Формат совместим с Prometheus text exposition format.

_METRICS = {
    "api_requests_total": {"type": "counter", "help": "Total API requests", "labels": ["method", "endpoint", "status"]},
    "api_request_duration_seconds": {"type": "histogram", "help": "Request duration", "labels": ["method", "endpoint"]},
    "api_predict_count": {"type": "counter", "help": "Total predictions made"},
    "api_model_loaded": {"type": "gauge", "help": "Whether the model is loaded (1=yes, 0=no)"},
}

_counters: dict[str, dict[tuple[str, ...], int]] = {
    "api_requests_total": {},
    "api_predict_count": {},
}
_gauge_values: dict[str, float] = {"api_model_loaded": 0.0}
_histogram_values: dict[str, dict[tuple[str, ...], list[float]]] = {
    "api_request_duration_seconds": {},
}


def _counter_key(labels: dict[str, str]) -> tuple[str, ...]:
    return tuple(labels.values())


def inc_counter(name: str, labels: dict[str, str], value: int = 1) -> None:
    key = _counter_key(labels)
    _counters[name][key] = _counters[name].get(key, 0) + value


def observe_histogram(name: str, labels: dict[str, str], value: float) -> None:
    key = _counter_key(labels)
    _histogram_values[name].setdefault(key, []).append(value)


def set_gauge(name: str, value: float) -> None:
    _gauge_values[name] = value


def generate_prometheus_metrics() -> str:
    """Генерирует текст в Prometheus exposition format."""
    lines: list[str] = []

    for metric_name, meta in _METRICS.items():
        lines.append(f"# HELP {metric_name} {meta['help']}")
        lines.append(f"# TYPE {metric_name} {meta['type']}")

        if meta["type"] in ("counter", "gauge"):
            if metric_name in _counters:
                for key, val in _counters[metric_name].items():
                    label_parts = ",".join(
                        f'{name}="{value}"'
                        for name, value in zip(meta.get("labels", []), key)
                    )
                    label_str = f"{{{label_parts}}}" if label_parts else ""
                    lines.append(f"{metric_name}{label_str} {val}")
            elif metric_name in _gauge_values:
                lines.append(f"{metric_name} {_gauge_values[metric_name]}")

        elif meta["type"] == "histogram":
            for key, values in _histogram_values.get(metric_name, {}).items():
                label_parts = ",".join(
                    f'{name}="{value}"'
                    for name, value in zip(meta.get("labels", []), key)
                )
                label_str = f"{{{label_parts}}}" if label_parts else ""

                if not values:
                    continue
                values_sorted = sorted(values)
                count = len(values_sorted)
                total_sum = sum(values_sorted)

                lines.append(f"{metric_name}_count{label_str} {count}")
                lines.append(f"{metric_name}_sum{label_str} {total_sum:.6f}")

    lines.append("")
    return "\n".join(lines)


# --- Модель ---
model = None
class_names: list[str] = []
device = str(resolve_device())


@app.on_event("startup")
async def load_model():
    global model, class_names
    ckpt_path = "outputs/best_model.pth"
    if Path(ckpt_path).exists():
        log.info("Загрузка модели из %s на %s...", ckpt_path, device)
        model, class_names, _ = load_model_from_checkpoint(ckpt_path, device=device)
        set_gauge("api_model_loaded", 1.0)
        log.info("Модель загружена. Классов: %d", len(class_names))
    else:
        log.warning("Чекпоинт %s не найден. Модель не загружена.", ckpt_path)
        set_gauge("api_model_loaded", 0.0)


# --- Middleware для сбора метрик ---


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)

    inc_counter("api_requests_total", {"method": method, "endpoint": endpoint, "status": status})
    observe_histogram("api_request_duration_seconds", {"method": method, "endpoint": endpoint}, duration)

    return response


# --- Эндпоинты ---


@app.get("/health")
async def health_check():
    """Эндпоинт для проверки работоспособности сервиса."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-метрики в text exposition формате."""
    return PlainTextResponse(content=generate_prometheus_metrics(), media_type="text/plain")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает изображение и возвращает список найденных объектов с их масками и уверенностью.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Модель еще не загружена")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        log.info("Predict запрос: %s (%d байт)", file.filename, len(contents))
        _, prediction = predict_instances(image, model, device=device)
        detections = summarize_predictions(prediction, class_names)

        inc_counter("api_predict_count", {})

        return JSONResponse(content={
            "filename": file.filename,
            "detections": detections,
            "count": len(detections),
        })
    except Exception as exc:
        log.error("Ошибка при обработке %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {exc}")


if __name__ == "__main__":
    log.info("Запуск API на %s:%d", "0.0.0.0", 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
