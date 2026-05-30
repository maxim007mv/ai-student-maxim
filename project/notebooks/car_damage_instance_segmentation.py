# %%
# Ноутбук-скрипт для академического проекта по instance segmentation.
# Его можно запускать по ячейкам в Jupyter / VS Code, либо целиком как обычный `.py` файл.

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from car_damage_segmentation.data import build_dataloaders, prepare_hf_car_dataset
from car_damage_segmentation.engine import fit
from car_damage_segmentation.inference import predict_and_visualize
from car_damage_segmentation.modeling import get_instance_segmentation_model, load_model_from_checkpoint
from car_damage_segmentation.utils import resolve_device, save_json, set_seed


# %%
# Базовая конфигурация эксперимента.
# При необходимости значения ниже можно менять прямо в ноутбуке.
CONFIG = {
    "dataset_id": "DrBimmer/car-parts-and-damage-dataset",
    "dataset_root": "data",
    "output_dir": "outputs",
    "epochs": 10,
    "batch_size": 2,
    "num_workers": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "val_size": 0.2,
    "seed": 42,
    "score_threshold": 0.5,
}

set_seed(CONFIG["seed"])
device = resolve_device()
device


# %%
# Скачивание и подготовка датасета с Hugging Face.
# На этом шаге:
# 1. исходный репозиторий датасета скачивается локально;
# 2. polygon-аннотации переводятся в единый manifest;
# 3. формируются train/val split и COCO JSON-файлы.
dataset_bundle = prepare_hf_car_dataset(
    dataset_id=CONFIG["dataset_id"],
    dataset_root=CONFIG["dataset_root"],
    val_size=CONFIG["val_size"],
    seed=CONFIG["seed"],
)
dataset_bundle


# %%
# Создание DataLoader.
# Train loader использует простые аугментации:
# - горизонтальный flip;
# - изменение яркости и контраста.
train_loader, val_loader = build_dataloaders(
    train_records_path=dataset_bundle.train_records_path,
    val_records_path=dataset_bundle.val_records_path,
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"],
)

print(f"Количество foreground-классов: {len(dataset_bundle.class_names)}")
print("Примеры классов:", dataset_bundle.class_names[:10])


# %%
# Инициализация предобученной модели Mask R-CNN и замена голов под наш датасет.
# +1 класс нужен для фонового класса background.
model = get_instance_segmentation_model(num_classes=len(dataset_bundle.class_names) + 1)
model.to(device)

optimizer = torch.optim.AdamW(
    params=[parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# %%
# Основной цикл обучения и базовой валидации.
# После каждой эпохи считаются:
# - validation loss;
# - mean IoU по маскам;
# - Precision и Recall при IoU >= 0.5.
history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=CONFIG["epochs"],
    checkpoint_dir=CONFIG["output_dir"],
    lr_scheduler=lr_scheduler,
    score_threshold=CONFIG["score_threshold"],
    class_names=dataset_bundle.class_names,
    categories=dataset_bundle.categories,
)

save_json(Path(CONFIG["output_dir"]) / "history.json", history)
history[-1]


# %%
# Загрузка лучшего чекпоинта и запуск инференса на одном изображении из validation-набора.
best_checkpoint = Path(CONFIG["output_dir"]) / "best_model.pth"
best_model, class_names, _ = load_model_from_checkpoint(best_checkpoint, device=str(device))

sample_record = train_loader.dataset.records[0]
result = predict_and_visualize(
    image_tensor=sample_record["image_path"],
    model=best_model,
    class_names=class_names,
    score_threshold=CONFIG["score_threshold"],
    device=str(device),
    show=True,
    return_data=True,
)

result["detections"][:5]


# %%
# Если нужно, итоговое изображение можно сохранить локально.
import cv2

output_image_path = Path(CONFIG["output_dir"]) / "sample_prediction.png"
cv2.imwrite(
    str(output_image_path),
    cv2.cvtColor(result["rendered_image"], cv2.COLOR_RGB2BGR),
)
print(f"Сохранено: {output_image_path}")
