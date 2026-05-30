from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .metrics import MetricAccumulator, accumulate_batch_metrics
from .utils import ensure_dir, log


def move_targets_to_device(
    targets: tuple[dict[str, torch.Tensor], ...],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Переносит все target-тензоры на нужное устройство."""
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> float:
    """Одна эпоха обучения модели instance segmentation."""
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Train", leave=False)
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

        loss_value = float(total_loss.detach().cpu().item())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Обнаружен нечисловой loss: {loss_value:.4f}")

        if scaler is not None and use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        running_loss += loss_value
        progress_bar.set_postfix(loss=f"{loss_value:.4f}")

    return running_loss / max(len(data_loader), 1)


def compute_validation_loss(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """
    Для torchvision detection API loss считается только в режиме `train`.
    Поэтому для validation-loss мы временно включаем train mode, но без градиентов.
    """
    previous_mode = model.training
    model.train()

    losses: list[float] = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Val loss", leave=False):
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)

            with autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
            losses.append(float(total_loss.detach().cpu().item()))

    model.train(previous_mode)
    return sum(losses) / max(len(losses), 1)


def validate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    score_threshold: float = 0.5,
    detection_iou_threshold: float = 0.5,
    mask_iou_threshold: float = 0.5,
    use_amp: bool = False,
) -> dict[str, float]:
    """Запускает валидацию: считает val loss и метрики сегментации/детекции."""
    val_loss = compute_validation_loss(model, data_loader, device=device, use_amp=use_amp)

    model.eval()
    accumulator = MetricAccumulator()

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Val metrics", leave=False):
            images_device = [image.to(device) for image in images]
            predictions = model(images_device)

            predictions_cpu = [{key: value.detach().cpu() for key, value in prediction.items()} for prediction in predictions]
            targets_cpu = [{key: value.detach().cpu() for key, value in target.items()} for target in targets]

            accumulate_batch_metrics(
                predictions=predictions_cpu,
                targets=targets_cpu,
                accumulator=accumulator,
                score_threshold=score_threshold,
                detection_iou_threshold=detection_iou_threshold,
                mask_iou_threshold=mask_iou_threshold,
            )

    metrics = accumulator.to_dict()
    metrics["val_loss"] = val_loss
    return metrics


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: str | Path = "outputs",
    lr_scheduler: Optional[Any] = None,
    score_threshold: float = 0.5,
    detection_iou_threshold: float = 0.5,
    mask_iou_threshold: float = 0.5,
    class_names: Optional[list[str]] = None,
    categories: Optional[list[dict[str, Any]]] = None,
    start_epoch: int = 0,
    history: Optional[list[dict[str, float]]] = None,
) -> list[dict[str, float]]:
    """Полный цикл обучения с сохранением лучшего чекпоинта по `mean_iou`."""
    checkpoint_dir = ensure_dir(checkpoint_dir)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    history = history or []
    best_mean_iou = max([h["mean_iou"] for h in history]) if history else -1.0

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        metrics = validate(
            model=model,
            data_loader=val_loader,
            device=device,
            score_threshold=score_threshold,
            detection_iou_threshold=detection_iou_threshold,
            mask_iou_threshold=mask_iou_threshold,
            use_amp=use_amp,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            **metrics,
        }
        history.append(epoch_record)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "class_names": class_names or [],
            "categories": categories or [],
        }
        torch.save(checkpoint_payload, checkpoint_dir / "last_model.pth")

        if metrics["mean_iou"] > best_mean_iou:
            best_mean_iou = metrics["mean_iou"]
            torch.save(checkpoint_payload, checkpoint_dir / "best_model.pth")

        log.info(
            "[Epoch %02d/%02d] "
            "train_loss=%.4f | val_loss=%.4f | mIoU=%.4f | precision=%.4f | recall=%.4f",
            epoch, num_epochs,
            train_loss,
            metrics["val_loss"],
            metrics["mean_iou"],
            metrics["precision"],
            metrics["recall"],
        )

    return history
