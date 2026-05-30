from __future__ import annotations

from pathlib import Path
from datasets import load_dataset
from .utils import ensure_dir, save_json
import cv2
import os

def prepare_cardd(output_root: str | Path = "data/cardd"):
    """
    Интеграция датасета CarDD (Xinkuang/CarDD).
    Этот скрипт скачивает датасет и готовит его в формате, совместимом с нашим проектом.
    """
    output_root = ensure_dir(output_root)
    img_dir = ensure_dir(output_root / "images")
    
    print("Загрузка CarDD с Hugging Face...")
    # Загружаем только subset 'instance_segmentation' если он есть, иначе весь
    ds = load_dataset("Xinkuang/CarDD", trust_remote_code=True)
    
    records = []
    
    # Маппинг классов CarDD (примерный, нужно уточнить по метаданным CarDD)
    # CarDD обычно имеет: Dent, Scratch, Crack, Glass Shatter, Lamp Broken, Tire Flat
    
    for split in ["train", "test"]:
        print(f"Обработка сплита {split}...")
        for i, example in enumerate(ds[split]):
            img = example["image"]
            img_filename = f"{split}_{i}.jpg"
            img_path = img_dir / img_filename
            
            # Сохраняем изображение
            if not img_path.exists():
                img.save(img_path)
            
            # Собираем аннотации
            # В Hugging Face datasets формат зависит от конкретного датасета.
            # Обычно это 'lvis' или 'coco' формат внутри 'annotations'
            
            # Внимание: CarDD может требовать специфического парсинга.
            # Для курсовой мы добавим этот код как заготовку.
            
    print("Подготовка CarDD завершена (заготовка).")

if __name__ == "__main__":
    # prepare_cardd()
    pass
