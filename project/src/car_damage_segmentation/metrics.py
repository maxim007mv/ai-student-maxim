from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import box_iou


@dataclass
class MetricAccumulator:
    """
    Накопитель метрик по всему validation-набору.
    `matched_mask_iou_sum / gt_instance_count` даёт mean IoU по GT-экземплярам,
    а пропущенные объекты автоматически вносят вклад 0.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_mask_iou_sum: float = 0.0
    gt_instance_count: int = 0

    def to_dict(self) -> dict[str, float]:
        precision_denominator = self.true_positives + self.false_positives
        recall_denominator = self.true_positives + self.false_negatives

        precision = self.true_positives / precision_denominator if precision_denominator > 0 else 0.0
        recall = self.true_positives / recall_denominator if recall_denominator > 0 else 0.0
        mean_iou = self.matched_mask_iou_sum / self.gt_instance_count if self.gt_instance_count > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "mean_iou": mean_iou,
            "tp": float(self.true_positives),
            "fp": float(self.false_positives),
            "fn": float(self.false_negatives),
        }


def filter_prediction_by_score(
    prediction: dict[str, torch.Tensor],
    score_threshold: float,
) -> dict[str, torch.Tensor]:
    """Оставляет только предсказания с вероятностью не ниже заданного порога."""
    if prediction.get("scores") is None or prediction["scores"].numel() == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "masks": torch.zeros((0, 1, 1), dtype=torch.uint8),
        }

    keep = prediction["scores"] >= score_threshold
    filtered = {
        "boxes": prediction["boxes"][keep].detach().cpu(),
        "labels": prediction["labels"][keep].detach().cpu(),
        "scores": prediction["scores"][keep].detach().cpu(),
        "masks": prediction["masks"][keep].detach().cpu(),
    }

    if filtered["masks"].ndim == 4:
        filtered["masks"] = (filtered["masks"][:, 0] >= 0.5).to(torch.uint8)
    return filtered


def pairwise_mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Считает попарные IoU между двумя наборами бинарных масок."""
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return torch.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=torch.float32)

    pred_flat = pred_masks.bool().flatten(1)
    gt_flat = gt_masks.bool().flatten(1)

    intersections = (pred_flat[:, None, :] & gt_flat[None, :, :]).sum(dim=-1).float()
    unions = (pred_flat[:, None, :] | gt_flat[None, :, :]).sum(dim=-1).float().clamp_min(1.0)
    return intersections / unions


def greedy_class_aware_match(
    iou_matrix: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
    """
    Жадное one-to-one сопоставление предсказаний и GT по IoU.
    Совпадение засчитывается только если классы одинаковы и IoU >= threshold.
    """
    if iou_matrix.numel() == 0:
        return []

    candidates: list[tuple[float, int, int]] = []
    for pred_index in range(iou_matrix.shape[0]):
        for gt_index in range(iou_matrix.shape[1]):
            if int(pred_labels[pred_index]) != int(gt_labels[gt_index]):
                continue
            iou_value = float(iou_matrix[pred_index, gt_index])
            if iou_value >= iou_threshold:
                candidates.append((iou_value, pred_index, gt_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    matches: list[tuple[float, int, int]] = []

    for iou_value, pred_index, gt_index in candidates:
        if pred_index in matched_predictions or gt_index in matched_targets:
            continue
        matched_predictions.add(pred_index)
        matched_targets.add(gt_index)
        matches.append((iou_value, pred_index, gt_index))

    return matches


def accumulate_batch_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    accumulator: MetricAccumulator,
    score_threshold: float = 0.5,
    detection_iou_threshold: float = 0.5,
    mask_iou_threshold: float = 0.5,
) -> MetricAccumulator:
    """Обновляет глобальный накопитель метрик статистикой по одному batch."""
    for prediction, target in zip(predictions, targets):
        pred = filter_prediction_by_score(prediction, score_threshold=score_threshold)
        gt_boxes = target["boxes"].detach().cpu()
        gt_labels = target["labels"].detach().cpu()
        gt_masks = target["masks"].detach().cpu().to(torch.uint8)

        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_masks = pred["masks"]

        accumulator.gt_instance_count += int(gt_labels.shape[0])

        detection_iou_matrix = box_iou(pred_boxes, gt_boxes) if len(pred_boxes) and len(gt_boxes) else torch.zeros(
            (len(pred_boxes), len(gt_boxes)),
            dtype=torch.float32,
        )
        detection_matches = greedy_class_aware_match(
            detection_iou_matrix,
            pred_labels,
            gt_labels,
            iou_threshold=detection_iou_threshold,
        )

        tp = len(detection_matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        accumulator.true_positives += tp
        accumulator.false_positives += fp
        accumulator.false_negatives += fn

        mask_iou_matrix = pairwise_mask_iou(pred_masks, gt_masks)
        mask_matches = greedy_class_aware_match(
            mask_iou_matrix,
            pred_labels,
            gt_labels,
            iou_threshold=mask_iou_threshold,
        )
        accumulator.matched_mask_iou_sum += sum(match[0] for match in mask_matches)

    return accumulator


def evaluate_predictions(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    score_threshold: float = 0.5,
    detection_iou_threshold: float = 0.5,
    mask_iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Удобная обёртка для расчёта метрик на произвольном списке предсказаний."""
    accumulator = MetricAccumulator()
    accumulate_batch_metrics(
        predictions=predictions,
        targets=targets,
        accumulator=accumulator,
        score_threshold=score_threshold,
        detection_iou_threshold=detection_iou_threshold,
        mask_iou_threshold=mask_iou_threshold,
    )
    return accumulator.to_dict()
