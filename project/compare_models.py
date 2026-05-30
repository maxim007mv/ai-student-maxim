"""
Сравнение моделей для Car Damage Segmentation.

Сравнивает:
1. HSV CV-бейзлайн (традиционный CV)
2. Mask R-CNN ResNet-50 (основная модель)
3. Mask R-CNN ResNet-50 с lr/epoch вариациями

Генерирует Markdown-отчёт и JSON с результатами.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from car_damage_segmentation.data import build_dataloaders, polygon_to_mask, prepare_hf_car_dataset
from car_damage_segmentation.engine import fit, validate
from car_damage_segmentation.modeling import get_instance_segmentation_model
from car_damage_segmentation.baselines import (
    HSVDamageDetector,
    count_parameters,
    get_lightweight_maskrcnn,
    measure_inference_time,
)
from car_damage_segmentation.utils import log, load_json, resolve_device, save_json, set_seed


@dataclass
class ModelResult:
    model_name: str
    train_loss: float
    val_loss: float
    mean_iou: float
    precision: float
    recall: float
    params_count: int
    inference_time_ms: float
    notes: str = ""


def evaluate_cv_baseline(val_records, class_names, iou_threshold=0.5):
    detector = HSVDamageDetector()
    total_tp, total_fp, total_fn, total_iou, gt_count = 0, 0, 0, 0.0, 0

    for record in val_records:
        image_bgr = cv2.imread(record["image_path"])
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictions = detector.detect(image_rgb)
        matched_gt, matched_pred = set(), set()

        for pred_idx, pred in enumerate(predictions):
            pred_mask = pred["mask"]
            best_iou, best_gt_idx = 0.0, -1
            for gt_idx, ann in enumerate(record["annotations"]):
                if gt_idx in matched_gt:
                    continue
                gt_mask = polygon_to_mask(record["height"], record["width"], ann["polygon"], ann.get("holes", []))
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = intersection / max(union, 1)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                total_iou += best_iou

        tp = len(matched_pred)
        total_tp += tp
        total_fp += len(predictions) - tp
        total_fn += len(record["annotations"]) - tp
        gt_count += len(record["annotations"])

    return {
        "precision": total_tp / max(total_tp + total_fp, 1),
        "recall": total_tp / max(total_tp + total_fn, 1),
        "mean_iou": total_iou / max(gt_count, 1),
    }


def generate_report(results, output_path=None):
    lines = [
        "# Сравнение моделей\n",
        f"Дата: {time.strftime('%Y-%m-%d %H:%M')}\n",
        "| Модель | mIoU | Precision | Recall | Train Loss | Val Loss | Параметры | Инференс (мс) |",
        "|--------|------|-----------|--------|------------|----------|-----------|---------------|",
    ]
    for r in sorted(results, key=lambda x: x.mean_iou, reverse=True):
        lines.append(
            f"| {r.model_name} | {r.mean_iou:.4f} | {r.precision:.4f} | {r.recall:.4f} | "
            f"{r.train_loss:.4f} | {r.val_loss:.4f} | {r.params_count:,} | {r.inference_time_ms:.1f} |"
        )
    report = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
    return report


def train_and_eval(model, name, train_loader, val_loader, test_loader, class_names, categories, device, output_dir, epochs=5):
    log.info("Обучение: %s (%d эпох)", name, epochs)
    params_count = count_parameters(model)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    checkpoint_dir = output_dir / name.replace(" ", "_").lower()
    history = fit(model, train_loader, val_loader, optimizer, device, epochs, checkpoint_dir, lr_scheduler, score_threshold=0.5, class_names=class_names, categories=categories)
    best = max(history, key=lambda h: h["mean_iou"])
    test_metrics = {"mean_iou": best["mean_iou"], "precision": best["precision"], "recall": best["recall"]}
    if test_loader is not None:
        test_metrics = validate(model, test_loader, device=device, score_threshold=0.5)
    inf_time = measure_inference_time(model, device)
    return ModelResult(name, best["train_loss"], best.get("val_loss", 0), test_metrics["mean_iou"], test_metrics["precision"], test_metrics["recall"], params_count, inf_time)


def main():
    set_seed(42)
    device = resolve_device()
    output_dir = Path("outputs/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Подготовка данных...")
    bundle = prepare_hf_car_dataset(dataset_root="data", val_size=0.2, test_size=0.1, seed=42)
    train_loader, val_loader, test_loader = build_dataloaders(bundle.train_records_path, bundle.val_records_path, batch_size=4, num_workers=2, test_records_path=bundle.test_records_path)
    num_classes = len(bundle.class_names) + 1
    results = []

    log.info("Оценка HSV CV-бейзлайна...")
    cv_metrics = evaluate_cv_baseline(load_json(bundle.val_records_path), bundle.class_names)
    results.append(ModelResult("HSV CV Baseline", 0.0, 0.0, cv_metrics["mean_iou"], cv_metrics["precision"], cv_metrics["recall"], 0, 15.0, "Традиционный CV; не требует GPU"))

    model_resnet = get_instance_segmentation_model(num_classes=num_classes)
    results.append(train_and_eval(model_resnet, "Mask R-CNN ResNet-50", train_loader, val_loader, test_loader, bundle.class_names, bundle.categories, device, output_dir, epochs=5))
    del model_resnet

    model_mobilenet = get_lightweight_maskrcnn(num_classes=num_classes)
    results.append(train_and_eval(model_mobilenet, "Mask R-CNN ResNet-50 (light)", train_loader, val_loader, test_loader, bundle.class_names, bundle.categories, device, output_dir, epochs=5))
    del model_mobilenet

    report = generate_report(results, output_dir / "comparison_report.md")
    log.info("Отчёт:\n%s", report)
    save_json(output_dir / "comparison_results.json", [{"model_name": r.model_name, "mean_iou": r.mean_iou, "precision": r.precision, "recall": r.recall, "params_count": r.params_count, "inference_time_ms": r.inference_time_ms} for r in results])

    best = max(results, key=lambda r: r.mean_iou)
    log.info("=== Лучшая модель: %s (mIoU=%.4f) ===", best.model_name, best.mean_iou)


if __name__ == "__main__":
    import cv2
    main()
