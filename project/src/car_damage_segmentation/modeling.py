from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .utils import resolve_device


def get_instance_segmentation_model(num_classes: int) -> torch.nn.Module:
    """
    Загружает предобученный Mask R-CNN и заменяет головы классификации и сегментации
    под количество классов конкретного датасета.
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    return model


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    """Восстанавливает модель и имена классов из сохранённого чекпоинта."""
    device_obj = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device_obj)

    class_names = checkpoint.get("class_names", [])
    if not class_names and "categories" in checkpoint:
        class_names = [category["name"] for category in checkpoint["categories"]]

    model = get_instance_segmentation_model(num_classes=len(class_names) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model, class_names, checkpoint
