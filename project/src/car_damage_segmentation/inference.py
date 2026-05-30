from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .metrics import filter_prediction_by_score
from .utils import generate_color_palette, resolve_device


def _to_rgb_numpy(image_input: str | Path | np.ndarray | Image.Image | torch.Tensor) -> np.ndarray:
    """Приводит входное изображение к формату RGB numpy uint8."""
    if isinstance(image_input, (str, Path)):
        image_bgr = cv2.imread(str(image_input))
        if image_bgr is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {image_input}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_input.copy()
    elif torch.is_tensor(image_input):
        tensor = image_input.detach().cpu()
        if tensor.ndim != 3 or tensor.shape[0] not in {1, 3}:
            raise ValueError("Ожидается тензор изображения формата [C, H, W].")
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        image_rgb = tensor.permute(1, 2, 0).numpy()
        if image_rgb.max() <= 1.0:
            image_rgb = image_rgb * 255.0
        image_rgb = image_rgb.clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Неподдерживаемый тип входа: {type(image_input)!r}")

    return np.ascontiguousarray(image_rgb.astype(np.uint8))


def predict_instances(
    image_input: str | Path | np.ndarray | Image.Image | torch.Tensor,
    model: torch.nn.Module,
    score_threshold: float = 0.5,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    """Запускает инференс модели и возвращает RGB-изображение и отфильтрованные предсказания."""
    device_obj = resolve_device(device)
    image_rgb = _to_rgb_numpy(image_input)
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

    model.to(device_obj)
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor.to(device_obj)])[0]

    prediction_cpu = {key: value.detach().cpu() for key, value in prediction.items()}
    prediction_cpu = filter_prediction_by_score(prediction_cpu, score_threshold=score_threshold)
    return image_rgb, prediction_cpu


def summarize_predictions(
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Готовит табличное представление результатов инференса для UI и логов."""
    summary: list[dict[str, Any]] = []
    for index in range(len(prediction["labels"])):
        label_id = int(prediction["labels"][index].item())
        label_name = class_names[label_id - 1]
        score = float(prediction["scores"][index].item())
        mask_area = int(prediction["masks"][index].sum().item())
        group = "Повреждение" if any(keyword in label_name.lower() for keyword in ("dent", "scratch", "broken", "crack", "shatter", "flat")) else "Деталь"
        summary.append(
            {
                "Класс": label_name,
                "Тип": group,
                "Уверенность": round(score, 4),
                "Площадь маски, px": mask_area,
            }
        )
    return summary


def draw_predictions(
    image_rgb: np.ndarray,
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
    alpha: float = 0.45,
) -> np.ndarray:
    """Накладывает маски, контуры, боксы и подписи на исходное изображение."""
    rendered = image_rgb.astype(np.float32).copy()
    palette = generate_color_palette(class_names)

    for index in range(len(prediction["labels"])):
        label_id = int(prediction["labels"][index].item())
        label_name = class_names[label_id - 1]
        color = np.array(palette[label_name], dtype=np.float32)
        mask = prediction["masks"][index].numpy().astype(bool)

        rendered[mask] = rendered[mask] * (1.0 - alpha) + color * alpha

    rendered = rendered.clip(0, 255).astype(np.uint8)

    for index in range(len(prediction["labels"])):
        label_id = int(prediction["labels"][index].item())
        label_name = class_names[label_id - 1]
        score = float(prediction["scores"][index].item())
        color = tuple(int(channel) for channel in palette[label_name])
        mask = prediction["masks"][index].numpy().astype(np.uint8)
        box = prediction["boxes"][index].numpy().astype(int).tolist()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, contourIdx=-1, color=color, thickness=2)

        x_min, y_min, x_max, y_max = box
        cv2.rectangle(rendered, (x_min, y_min), (x_max, y_max), color, 2)

        label_text = f"{label_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        text_y = max(y_min - 8, text_height + 8)
        cv2.rectangle(
            rendered,
            (x_min, text_y - text_height - baseline - 6),
            (x_min + text_width + 8, text_y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            rendered,
            label_text,
            (x_min + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return rendered


def predict_and_visualize(
    image_tensor: str | Path | np.ndarray | Image.Image | torch.Tensor,
    model: torch.nn.Module,
    class_names: list[str],
    score_threshold: float = 0.5,
    alpha: float = 0.45,
    device: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    show: bool = True,
    return_data: bool = False,
) -> Any:
    """
    Главная функция инференса и визуализации.
    Она:
    1. прогоняет изображение через модель;
    2. накладывает маски и bbox;
    3. по желанию возвращает figure, отрисованную картинку и таблицу детекций.
    """
    image_rgb, prediction = predict_instances(
        image_input=image_tensor,
        model=model,
        score_threshold=score_threshold,
        device=device,
    )
    rendered = draw_predictions(
        image_rgb=image_rgb,
        prediction=prediction,
        class_names=class_names,
        alpha=alpha,
    )
    detections = summarize_predictions(prediction, class_names=class_names)

    figure = None
    if show:
        figure, axis = plt.subplots(figsize=figsize)
        axis.imshow(rendered)
        axis.set_title("Instance Segmentation: детали автомобиля и повреждения")
        axis.axis("off")
        plt.tight_layout()

    if return_data:
        return {
            "figure": figure,
            "rendered_image": rendered,
            "prediction": prediction,
            "detections": detections,
        }
    return figure if show else rendered
