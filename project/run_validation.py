import torch
import cv2
import numpy as np
from pathlib import Path
from car_damage_segmentation.modeling import load_model_from_checkpoint
from car_damage_segmentation.inference import predict_and_visualize
from car_damage_segmentation.utils import load_json, resolve_device
import sys

def run_system_validation(checkpoint_path="outputs/best_model.pth", num_samples=5):
    device = str(resolve_device())
    if not Path(checkpoint_path).exists():
        print(f"Ошибка: Чекпоинт {checkpoint_path} не найден.")
        return

    print(f"Загрузка модели из {checkpoint_path}...")
    model, class_names, _ = load_model_from_checkpoint(checkpoint_path, device=device)
    
    val_records_path = "data/processed/val_records.json"
    if not Path(val_records_path).exists():
        print("Ошибка: Данные валидации не найдены. Запустите сначала обучение или подготовку данных.")
        return
        
    val_records = load_json(val_records_path)
    samples = val_records[:num_samples]
    
    output_dir = Path("outputs/validation_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Запуск валидации на {len(samples)} примерах...")
    for i, record in enumerate(samples):
        img_path = record["image_path"]
        print(f"Обработка {img_path}...")
        
        result = predict_and_visualize(
            image_tensor=img_path,
            model=model,
            class_names=class_names,
            score_threshold=0.5,
            device=device,
            show=False,
            return_data=True
        )
        
        rendered = result["rendered_image"]
        # Сохраняем в BGR для OpenCV
        cv2.imwrite(str(output_dir / f"val_result_{i}.jpg"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        
    print(f"Валидация завершена. Результаты сохранены в {output_dir}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PROJECT_ROOT / "src"))
    run_system_validation()
