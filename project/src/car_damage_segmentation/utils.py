from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def setup_logger(
    name: str = "car_damage_segmentation",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Настраивает структурированное логирование.
    Пишет в консоль и опционально в файл.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(console_fmt)
        logger.addHandler(file_handler)

    return logger


# Глобальный логгер проекта
log = setup_logger()


def set_seed(seed: int = 42) -> None:
    """Фиксирует генераторы случайных чисел для воспроизводимости эксперимента."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info("Random seed установлен: %d", seed)


def ensure_dir(path: str | Path) -> Path:
    """Создаёт директорию, если она ещё не существует."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: Any) -> None:
    """Сохраняет Python-объект в JSON с отступами для удобного чтения."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Загружает JSON-файл и возвращает Python-объект."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple:
    """Специальная функция колляции для detection/segmentation моделей torchvision."""
    return tuple(zip(*batch))


def resolve_device(device: str | None = None) -> torch.device:
    """Выбирает GPU, если он доступен, иначе работает на CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_color_palette(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """
    Строит детерминированную палитру цветов для визуализации классов.
    Цвета генерируются через простой хэш имени класса.
    """
    palette: dict[str, tuple[int, int, int]] = {}
    for class_name in class_names:
        base = abs(hash(class_name))
        color = (
            70 + (base % 160),
            70 + ((base // 7) % 160),
            70 + ((base // 17) % 160),
        )
        palette[class_name] = color
    return palette
