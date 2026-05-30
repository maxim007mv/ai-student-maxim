from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from datasets import Dataset as HFDataset
from huggingface_hub import snapshot_download
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

from .utils import collate_fn, ensure_dir, load_json, save_json

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DAMAGE_KEYWORDS = (
    "dent",
    "scratch",
    "broken",
    "crack",
    "shatter",
    "flat",
    "damage",
)


@dataclass
class PreparedDatasetBundle:
    dataset_root: Path
    raw_repo_dir: Path
    processed_dir: Path
    all_records_path: Path
    train_records_path: Path
    val_records_path: Path
    test_records_path: Path
    labels_path: Path
    coco_all_path: Path
    coco_train_path: Path
    coco_val_path: Path
    coco_test_path: Path
    class_names: list[str]
    categories: list[dict[str, Any]]


def infer_supercategory(class_name: str) -> str:
    """Эвристически делит классы на автомобильные детали и повреждения."""
    class_name_lower = class_name.lower()
    if any(keyword in class_name_lower for keyword in DAMAGE_KEYWORDS):
        return "damage"
    return "part"


def flatten_polygon(points: list[list[float]]) -> list[float]:
    """Переводит список точек [[x, y], ...] в COCO-совместимый плоский список."""
    flattened: list[float] = []
    for x_coord, y_coord in points:
        flattened.extend([float(x_coord), float(y_coord)])
    return flattened


def polygon_to_mask(
    height: int,
    width: int,
    polygon: list[float],
    holes: Optional[list[list[float]]] = None,
) -> np.ndarray:
    """
    Конвертирует polygon-аннотацию в бинарную маску.
    Внешний контур кодируется через pycocotools, внутренние контуры вычитаются из маски.
    """
    if len(polygon) < 6:
        return np.zeros((height, width), dtype=np.uint8)

    rle = mask_utils.frPyObjects([polygon], height, width)
    decoded = mask_utils.decode(rle)
    mask = decoded if decoded.ndim == 2 else np.any(decoded, axis=2)
    mask = mask.astype(np.uint8)

    for hole in holes or []:
        if len(hole) < 6:
            continue
        hole_rle = mask_utils.frPyObjects([hole], height, width)
        hole_decoded = mask_utils.decode(hole_rle)
        hole_mask = hole_decoded if hole_decoded.ndim == 2 else np.any(hole_decoded, axis=2)
        mask = np.where(hole_mask, 0, mask).astype(np.uint8)

    return mask


def mask_to_box(mask: np.ndarray) -> Optional[list[float]]:
    """Находит ограничивающий прямоугольник по бинарной маске."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


def download_hf_dataset(
    dataset_id: str = "DrBimmer/car-parts-and-damage-dataset",
    download_dir: str | Path = "data/raw/car-parts-and-damage-dataset",
) -> Path:
    """
    Скачивает датасет с Hugging Face Hub локально.
    Используется `snapshot_download`, чтобы получить все исходные изображения и JSON-аннотации.
    """
    download_dir = ensure_dir(download_dir)
    if any(download_dir.rglob("ann/*.json")) and any(download_dir.rglob("img/*")):
        return Path(download_dir)

    snapshot_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(download_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.webp",
        ],
    )
    return Path(snapshot_path)


def _find_matching_image(image_dir: Path, annotation_file_name: str) -> Optional[Path]:
    """Подбирает изображение, соответствующее JSON-аннотации."""
    expected_name = annotation_file_name.removesuffix(".json")
    candidate = image_dir / expected_name
    if candidate.exists():
        return candidate

    stem = Path(expected_name).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _parse_annotation_file(annotation_path: Path, image_path: Path, image_id: int) -> dict[str, Any]:
    """Читает один JSON-файл и собирает все polygon-объекты для изображения."""
    payload = load_json(annotation_path)
    size_info = payload.get("size", {})
    width = int(size_info.get("width", 0))
    height = int(size_info.get("height", 0))

    if width <= 0 or height <= 0:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {image_path}")
        height, width = image.shape[:2]

    annotations: list[dict[str, Any]] = []
    for obj in payload.get("objects", []):
        if obj.get("geometryType") != "polygon":
            continue

        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) < 3:
            continue

        category_name = str(obj.get("classTitle", "")).strip()
        if not category_name:
            continue

        polygon = flatten_polygon(exterior)
        if len(polygon) < 6:
            continue

        holes: list[list[float]] = []
        for interior in obj.get("points", {}).get("interior", []):
            if len(interior) >= 3:
                holes.append(flatten_polygon(interior))

        annotations.append(
            {
                "source_annotation_id": int(obj.get("id", -1)),
                "category_name": category_name,
                "polygon": polygon,
                "holes": holes,
            }
        )

    return {
        "image_id": image_id,
        "image_path": str(image_path.resolve()),
        "width": width,
        "height": height,
        "annotations": annotations,
    }


def build_manifest_from_repo(repo_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Обходит исходный репозиторий датасета и собирает единый manifest.
    Каждое изображение получает список экземпляров с polygon-аннотациями.
    """
    repo_dir = Path(repo_dir)
    annotation_files = sorted(repo_dir.rglob("ann/*.json"))
    discovered_categories: dict[str, dict[str, str]] = {}
    records: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for annotation_path in annotation_files:
        image_dir = annotation_path.parent.parent / "img"
        image_path = _find_matching_image(image_dir, annotation_path.name)
        if image_path is None:
            continue

        record = _parse_annotation_file(annotation_path, image_path, next_image_id)
        if not record["annotations"]:
            continue

        for annotation in record["annotations"]:
            category_name = annotation["category_name"]
            if not category_name:
                continue

            discovered_categories.setdefault(
                category_name,
                {
                    "name": category_name,
                    "supercategory": infer_supercategory(category_name),
                },
            )
            annotation["annotation_id"] = next_annotation_id
            next_annotation_id += 1

        record["source_annotation_file"] = str(annotation_path.resolve())
        records.append(record)
        next_image_id += 1

    class_names = sorted(discovered_categories.keys())
    categories = [
        {
            "id": category_id,
            "name": class_name,
            "supercategory": discovered_categories[class_name]["supercategory"],
        }
        for category_id, class_name in enumerate(class_names, start=1)
    ]
    category_to_id = {category["name"]: category["id"] for category in categories}

    for record in records:
        for annotation in record["annotations"]:
            annotation["category_id"] = category_to_id[annotation["category_name"]]

    return records, categories, class_names


def export_records_to_coco(
    records: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Экспортирует подготовленный manifest в стандартный COCO JSON.
    Это удобно для отладки, визуализации и совместимости с внешними инструментами.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for record in records:
        images.append(
            {
                "id": record["image_id"],
                "file_name": record["image_path"],
                "width": record["width"],
                "height": record["height"],
            }
        )

        for annotation in record["annotations"]:
            mask = polygon_to_mask(
                height=record["height"],
                width=record["width"],
                polygon=annotation["polygon"],
                holes=annotation.get("holes", []),
            )
            box = mask_to_box(mask)
            if box is None:
                continue

            x_min, y_min, x_max, y_max = box
            annotations.append(
                {
                    "id": annotation["annotation_id"],
                    "image_id": record["image_id"],
                    "category_id": annotation["category_id"],
                    "segmentation": [annotation["polygon"]],
                    "area": int(mask.sum()),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "iscrowd": 0,
                }
            )

    save_json(
        output_path,
        {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        },
    )


def prepare_hf_car_dataset(
    dataset_id: str = "DrBimmer/car-parts-and-damage-dataset",
    dataset_root: str | Path = "data",
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
) -> PreparedDatasetBundle:
    """
    Полный цикл подготовки:
    1. скачивание датасета с Hugging Face;
    2. построение общего manifest;
    3. разбиение train/validation/test через библиотеку datasets;
    4. сохранение COCO JSON и служебных файлов.
    """
    from .utils import log

    dataset_root = ensure_dir(dataset_root)
    raw_repo_dir = dataset_root / "raw" / dataset_id.replace("/", "__")
    processed_dir = ensure_dir(dataset_root / "processed")

    log.info("Скачивание датасета %s ...", dataset_id)
    raw_repo_dir = download_hf_dataset(dataset_id=dataset_id, download_dir=raw_repo_dir)
    log.info("Построение manifest из %s ...", raw_repo_dir)
    records, categories, class_names = build_manifest_from_repo(raw_repo_dir)
    if not records:
        raise RuntimeError("Не удалось собрать ни одной аннотированной записи из датасета.")
    log.info("Собрано %d записей, %d классов", len(records), len(class_names))

    hf_dataset = HFDataset.from_list(records)

    # Первый сплит: отделяем test (hold-out)
    if test_size > 0:
        split1 = hf_dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        train_val_records = [split1["train"][i] for i in range(len(split1["train"]))]
        test_records = [split1["test"][i] for i in range(len(split1["test"]))]
        log.info("Test-holdout: %d записей (%.1f%%)", len(test_records), test_size * 100)
    else:
        train_val_records = records
        test_records = []

    # Второй сплит: train / val из оставшихся
    adjusted_val_ratio = val_size / (1.0 - test_size) if test_size > 0 else val_size
    train_val_dataset = HFDataset.from_list(train_val_records)
    split2 = train_val_dataset.train_test_split(
        test_size=adjusted_val_ratio, seed=seed, shuffle=True
    )
    train_records = [split2["train"][i] for i in range(len(split2["train"]))]
    val_records = [split2["test"][i] for i in range(len(split2["test"]))]
    log.info(
        "Train: %d | Val: %d | Test: %d",
        len(train_records), len(val_records), len(test_records),
    )

    labels_payload = {
        "background": "__background__",
        "classes": categories,
    }

    all_records_path = processed_dir / "all_records.json"
    train_records_path = processed_dir / "train_records.json"
    val_records_path = processed_dir / "val_records.json"
    test_records_path = processed_dir / "test_records.json"
    labels_path = processed_dir / "labels.json"
    coco_all_path = processed_dir / "all_instances_coco.json"
    coco_train_path = processed_dir / "train_instances_coco.json"
    coco_val_path = processed_dir / "val_instances_coco.json"
    coco_test_path = processed_dir / "test_instances_coco.json"

    save_json(all_records_path, records)
    save_json(train_records_path, train_records)
    save_json(val_records_path, val_records)
    save_json(test_records_path, test_records)
    save_json(labels_path, labels_payload)
    export_records_to_coco(records, categories, coco_all_path)
    export_records_to_coco(train_records, categories, coco_train_path)
    export_records_to_coco(val_records, categories, coco_val_path)
    if test_records:
        export_records_to_coco(test_records, categories, coco_test_path)

    return PreparedDatasetBundle(
        dataset_root=dataset_root,
        raw_repo_dir=raw_repo_dir,
        processed_dir=processed_dir,
        all_records_path=all_records_path,
        train_records_path=train_records_path,
        val_records_path=val_records_path,
        test_records_path=test_records_path,
        labels_path=labels_path,
        coco_all_path=coco_all_path,
        coco_train_path=coco_train_path,
        coco_val_path=coco_val_path,
        coco_test_path=coco_test_path,
        class_names=class_names,
        categories=categories,
    )


def get_train_transforms() -> A.Compose:
    """Аугментации для train: горизонтальный flip и изменение яркости/контраста."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )


def get_val_transforms() -> A.Compose:
    """Для validation обычно аугментации не применяются, только единый интерфейс вызова."""
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )


class CarDamageDataset(Dataset):
    """
    Кастомный Dataset для instance segmentation.
    На выходе возвращает:
    - image: тензор [C, H, W] в диапазоне [0, 1];
    - target: словарь в формате, который ожидает torchvision detection API.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.records = records
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image_path = Path(record["image_path"])

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []

        for annotation in record["annotations"]:
            mask = polygon_to_mask(
                height=height,
                width=width,
                polygon=annotation["polygon"],
                holes=annotation.get("holes", []),
            )
            if mask.sum() == 0:
                continue

            box = mask_to_box(mask)
            if box is None:
                continue

            boxes.append(box)
            labels.append(int(annotation["category_id"]))
            masks.append(mask.astype(np.uint8))

        if self.transforms is not None:
            transformed = self.transforms(
                image=image_rgb,
                masks=masks,
                bboxes=boxes,
                class_labels=labels,
            )
            image_rgb = transformed["image"]
            masks = [np.asarray(mask, dtype=np.uint8) for mask in transformed["masks"]]
            boxes = [list(map(float, box)) for box in transformed["bboxes"]]
            labels = [int(label) for label in transformed["class_labels"]]

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float() / 255.0

        if masks:
            masks_array = np.stack(masks, axis=0)
        else:
            masks_array = np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

        boxes_array = (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels_array = np.asarray(labels, dtype=np.int64)
        areas_array = (
            masks_array.reshape(masks_array.shape[0], -1).sum(axis=1).astype(np.float32)
            if masks_array.shape[0] > 0
            else np.zeros((0,), dtype=np.float32)
        )

        target = {
            "boxes": torch.as_tensor(boxes_array, dtype=torch.float32),
            "labels": torch.as_tensor(labels_array, dtype=torch.int64),
            "masks": torch.as_tensor(masks_array, dtype=torch.uint8),
            "image_id": torch.tensor([int(record["image_id"])], dtype=torch.int64),
            "area": torch.as_tensor(areas_array, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels_array),), dtype=torch.int64),
        }

        return image_tensor, target


def build_dataloaders(
    train_records_path: str | Path,
    val_records_path: str | Path,
    batch_size: int = 2,
    num_workers: int = 2,
    test_records_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Собирает train/validation/test DataLoader поверх сохранённых JSON-records."""
    train_records = load_json(train_records_path)
    val_records = load_json(val_records_path)

    train_dataset = CarDamageDataset(train_records, transforms=get_train_transforms())
    val_dataset = CarDamageDataset(val_records, transforms=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    test_loader = None
    if test_records_path and Path(test_records_path).exists():
        test_records = load_json(test_records_path)
        if test_records:
            test_dataset = CarDamageDataset(test_records, transforms=get_val_transforms())
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

    return train_loader, val_loader, test_loader
