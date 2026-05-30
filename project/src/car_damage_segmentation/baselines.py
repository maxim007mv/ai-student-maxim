"""
Классические baseline-модели для сравнения с Mask R-CNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .data import polygon_to_mask

@dataclass
class HSVDamageDetector:
    damage_hsv_ranges: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = field(
        default_factory=lambda: {
            "dent": ((0, 5, 50), (25, 100, 255)),
            "scratch": ((0, 0, 180), (180, 30, 255)),
            "broken": ((0, 0, 30), (180, 50, 200)),
            "crack": ((0, 0, 20), (180, 40, 180)),
        }
    )
    morph_kernel_size: int = 3
    min_contour_area: int = 200

    def detect(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        height, width = image_rgb.shape[:2]
        detections: list[dict[str, Any]] = []

        for damage_type, (lower, upper) in self.damage_hsv_ranges.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area: continue
                x, y, w, h = cv2.boundingRect(contour)
                contour_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
                detections.append({
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "mask": contour_mask, "label": damage_type, "score": 1.0, "area": float(area),
                })
        return detections

def get_lightweight_maskrcnn(num_classes: int) -> torch.nn.Module:
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    import time
    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad(): _ = model([dummy_input[0]])
    return (time.perf_counter() - start) / 10 * 1000
