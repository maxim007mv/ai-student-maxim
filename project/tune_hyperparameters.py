"""
Подбор гиперпараметров с использованием Optuna.

Оптимизирует:
- learning rate
- batch size
- weight decay
- optimizer type
- backbone architecture

Целевая метрика: mean IoU на валидационной выборке.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from car_damage_segmentation.data import build_dataloaders, prepare_hf_car_dataset
from car_damage_segmentation.engine import fit
from car_damage_segmentation.modeling import get_instance_segmentation_model
from car_damage_segmentation.baselines import get_lightweight_maskrcnn
from car_damage_segmentation.utils import log, resolve_device, set_seed, setup_logger

# Проверяем доступность Optuna
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    log.warning("Optuna не установлена. Установите: pip install optuna")


DATASET_ID = "DrBimmer/car-parts-and-damage-dataset"
N_TRIALS = 20
N_EPOCHS_PER_TRIAL = 5


def prepare_data(val_size: float = 0.2, test_size: float = 0.1, seed: int = 42):
    """Подготавливает данные один раз для всех trials."""
    dataset_bundle = prepare_hf_car_dataset(
        dataset_id=DATASET_ID,
        dataset_root="data",
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    train_loader, val_loader, _ = build_dataloaders(
        train_records_path=dataset_bundle.train_records_path,
        val_records_path=dataset_bundle.val_records_path,
        batch_size=1,  # Будет переопределено в trial
        num_workers=2,
    )
    return dataset_bundle, train_loader, val_loader


def objective(trial: Trial) -> float:
    """Целевая функция Optuna: обучает модель с предложенными гиперпараметрами."""
    set_seed(42)
    device = resolve_device()

    # Гиперпараметры для подбора
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    backbone = trial.suggest_categorical("backbone", ["resnet50", "mobilenetv3"])

    log.info(
        "Trial %d: lr=%.6f, bs=%d, wd=%.6f, opt=%s, backbone=%s",
        trial.number, lr, batch_size, weight_decay, optimizer_name, backbone,
    )

    # Данные
    dataset_bundle = prepare_hf_car_dataset(
        dataset_id=DATASET_ID,
        dataset_root="data",
        val_size=0.25,
        test_size=0.0,
        seed=42,
    )
    train_loader, val_loader, _ = build_dataloaders(
        train_records_path=dataset_bundle.train_records_path,
        val_records_path=dataset_bundle.val_records_path,
        batch_size=batch_size,
        num_workers=2,
    )

    # Модель
    num_classes = len(dataset_bundle.class_names) + 1
    if backbone == "mobilenetv3":
        model = get_lightweight_maskrcnn(num_classes=num_classes)
    else:
        model = get_instance_segmentation_model(num_classes=num_classes)

    model.to(device)

    # Оптимизатор
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Обучение
    output_dir = Path("outputs/optuna_trials") / f"trial_{trial.number:03d}"
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=N_EPOCHS_PER_TRIAL,
        checkpoint_dir=output_dir,
        lr_scheduler=lr_scheduler,
        score_threshold=0.5,
    )

    # Возвращаем лучший mIoU
    best_miou = max(h["mean_iou"] for h in history) if history else 0.0
    log.info("Trial %d: best mIoU = %.4f", trial.number, best_miou)

    # Дополнительно логируем последние метрики
    trial.set_user_attr("final_train_loss", history[-1]["train_loss"])
    trial.set_user_attr("final_val_loss", history[-1]["val_loss"])
    trial.set_user_attr("final_precision", history[-1]["precision"])
    trial.set_user_attr("final_recall", history[-1]["recall"])

    return best_miou


def main():
    if not OPTUNA_AVAILABLE:
        log.error("Optuna не установлена. Выход.")
        return

    log.info("Запуск подбора гиперпараметров (Optuna)")

    study = optuna.create_study(
        direction="maximize",
        study_name="car-damage-hparam-tuning",
        storage="sqlite:///outputs/optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=3600 * 6)

    log.info("=== Лучшие гиперпараметры ===")
    for key, value in study.best_params.items():
        log.info("  %s = %s", key, value)
    log.info("Лучший mIoU: %.4f", study.best_value)

    # Сохраняем отчёт
    import json
    report_path = Path("outputs/hparam_tuning_report.json")
    report = {
        "best_params": study.best_params,
        "best_miou": study.best_value,
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
            if t.value is not None
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("Отчёт сохранён в %s", report_path)

    # Визуализация
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("outputs/optuna_history.png")
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("outputs/optuna_importances.png")
        log.info("Графики сохранены в outputs/")
    except Exception as exc:
        log.warning("Не удалось построить графики Optuna: %s", exc)


if __name__ == "__main__":
    main()
