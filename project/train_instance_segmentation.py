"""
Скрипт обучения модели instance segmentation с трекингом MLflow.

Поддерживает:
- Обучение Mask R-CNN (ResNet-50 или MobileNetV3)
- Трекинг экспериментов через MLflow
- Автоматическое логирование метрик, параметров и артефактов
- Resume из чекпоинта
- Оценку на test-выборке после обучения
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from car_damage_segmentation.data import build_dataloaders, prepare_hf_car_dataset
from car_damage_segmentation.engine import fit, validate
from car_damage_segmentation.modeling import get_instance_segmentation_model
from car_damage_segmentation.baselines import (
    count_parameters,
    get_lightweight_maskrcnn,
    measure_inference_time,
)
from car_damage_segmentation.utils import (
    log,
    resolve_device,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение Mask R-CNN для instance segmentation автомобильных деталей и повреждений.",
    )
    # Данные
    parser.add_argument("--dataset-id", type=str, default="DrBimmer/car-parts-and-damage-dataset")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    # Обучение
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для продолжения обучения")
    # Модель
    parser.add_argument(
        "--backbone", type=str, choices=("resnet50", "mobilenetv3"), default="resnet50",
        help="Бэкбон модели: resnet50 (точнее) или mobilenetv3 (быстрее)",
    )
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true", help="Отключить MLflow-трекинг")
    parser.add_argument("--experiment-name", type=str, default="car-damage-segmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Настройка MLflow
    mlflow_active = False
    if not args.no_mlflow:
        try:
            import mlflow

            mlflow.set_experiment(args.experiment_name)
            mlflow_active = True
            log.info("MLflow трекинг активирован. Experiment: %s", args.experiment_name)
        except ImportError:
            log.warning("MLflow не установлен. Установите: pip install mlflow")
        except Exception as exc:
            log.warning("Не удалось подключиться к MLflow серверу: %s", exc)

    if mlflow_active:
        mlflow.start_run()
        mlflow.log_params({
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "seed": args.seed,
        })

    # 1. Подготовка данных
    log.info("1/5. Скачивание и подготовка датасета...")
    dataset_bundle = prepare_hf_car_dataset(
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    log.info("2/5. Построение DataLoader...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_records_path=dataset_bundle.train_records_path,
        val_records_path=dataset_bundle.val_records_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_records_path=dataset_bundle.test_records_path,
    )

    # 3. Инициализация модели
    log.info("3/5. Инициализация модели (backbone: %s)...", args.backbone)
    num_classes = len(dataset_bundle.class_names) + 1

    if args.backbone == "mobilenetv3":
        model = get_lightweight_maskrcnn(num_classes=num_classes)
    else:
        model = get_instance_segmentation_model(num_classes=num_classes)

    params = count_parameters(model)
    log.info("Модель: %s | Параметров: %s | Классов: %d", args.backbone, f"{params:,}", num_classes)

    if mlflow_active:
        mlflow.log_params({"model_params": params, "num_classes": num_classes, "backbone": args.backbone})

    start_epoch = 0
    history: list[dict[str, float]] = []

    if args.resume and Path(args.resume).exists():
        log.info("Загрузка чекпоинта: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        history = checkpoint.get("history", [])
        log.info("Продолжаем с эпохи %d", start_epoch + 1)

    model.to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 4. Обучение
    log.info("4/5. Запуск обучения (%d эпох)...", args.epochs)
    updated_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=args.output_dir,
        lr_scheduler=lr_scheduler,
        score_threshold=args.score_threshold,
        class_names=dataset_bundle.class_names,
        categories=dataset_bundle.categories,
        start_epoch=start_epoch,
        history=history,
    )

    # Логируем метрики в MLflow
    for record in updated_history[-args.epochs:]:
        epoch_num = int(record["epoch"])
        if mlflow_active:
            mlflow.log_metrics({
                "train_loss": record["train_loss"],
                "val_loss": record["val_loss"],
                "mean_iou": record["mean_iou"],
                "precision": record["precision"],
                "recall": record["recall"],
            }, step=epoch_num)

    save_json(output_dir / "history.json", updated_history)

    # 5. Оценка на test-выборке
    log.info("5/5. Финальная оценка на test-выборке...")
    if test_loader is not None:
        test_metrics = validate(
            model=model,
            data_loader=test_loader,
            device=device,
            score_threshold=args.score_threshold,
        )
        log.info(
            "Test metrics | mIoU=%.4f | Precision=%.4f | Recall=%.4f",
            test_metrics["mean_iou"],
            test_metrics["precision"],
            test_metrics["recall"],
        )
        save_json(output_dir / "test_metrics.json", test_metrics)
        if mlflow_active:
            mlflow.log_metrics({
                "test_mean_iou": test_metrics["mean_iou"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
            })

    # Логируем время инференса
    inf_time = measure_inference_time(model, device)
    log.info("Среднее время инференса: %.1f мс", inf_time)
    if mlflow_active:
        mlflow.log_metrics({"inference_time_ms": inf_time})
        mlflow.log_artifact(str(output_dir / "history.json"))
        mlflow.end_run()

    log.info(
        "Обучение завершено. Лучший чекпоинт: %s",
        output_dir / "best_model.pth",
    )


if __name__ == "__main__":
    main()
