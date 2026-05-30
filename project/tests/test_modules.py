"""
Модульные тесты для пакета car_damage_segmentation.

Покрывают:
- data: парсинг аннотаций, конвертация полигонов, построение COCO
- metrics: IoU, метрики, greedy matching
- modeling: создание и загрузка модели
- inference: predict, визуализация
- baselines: HSV-детектор, сравнение моделей
- utils: save/load JSON, set_seed, resolve_device
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))


# Utils tests

class TestUtils:
    def test_set_seed(self):
        from car_damage_segmentation.utils import set_seed
        set_seed(42)
        a = torch.rand(10)
        set_seed(42)
        b = torch.rand(10)
        assert torch.equal(a, b), "set_seed не обеспечивает воспроизводимость"

    def test_resolve_device(self):
        from car_damage_segmentation.utils import resolve_device
        device = resolve_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "cpu")

    def test_save_load_json(self):
        from car_damage_segmentation.utils import save_json, load_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            data = {"key": "value", "list": [1, 2, 3]}
            save_json(path, data)
            assert path.exists()
            loaded = load_json(path)
            assert loaded == data

    def test_generate_color_palette(self):
        from car_damage_segmentation.utils import generate_color_palette
        names = ["door", "dent", "scratch"]
        palette = generate_color_palette(names)
        assert len(palette) == 3
        for name in names:
            assert name in palette
            color = palette[name]
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_ensure_dir(self):
        from car_damage_segmentation.utils import ensure_dir
        with tempfile.TemporaryDirectory() as tmp:
            p = ensure_dir(Path(tmp) / "sub" / "dir")
            assert p.exists()
            assert p.is_dir()


# Data tests

class TestData:
    def test_flatten_polygon(self):
        from car_damage_segmentation.data import flatten_polygon
        points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = flatten_polygon(points)
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_polygon_to_mask(self):
        from car_damage_segmentation.data import polygon_to_mask
        # Квадрат 50x50 в центре изображения 100x100
        poly = [25.0, 25.0, 75.0, 25.0, 75.0, 75.0, 25.0, 75.0]
        mask = polygon_to_mask(100, 100, poly)
        assert mask.shape == (100, 100)
        assert mask.sum() > 0
        assert mask.sum() < 100 * 100  # не всё изображение

    def test_polygon_to_mask_invalid(self):
        from car_damage_segmentation.data import polygon_to_mask
        # Всего 2 точки — невалидный полигон
        mask = polygon_to_mask(100, 100, [0.0, 0.0, 10.0, 10.0])
        assert mask.sum() == 0  # пустая маска

    def test_mask_to_box(self):
        from car_damage_segmentation.data import mask_to_box
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 1
        box = mask_to_box(mask)
        assert box is not None
        assert len(box) == 4
        x_min, y_min, x_max, y_max = box
        assert 28 <= x_min <= 32
        assert 18 <= y_min <= 22
        assert 78 <= x_max <= 82
        assert 58 <= y_max <= 62

    def test_mask_to_box_empty(self):
        from car_damage_segmentation.data import mask_to_box
        mask = np.zeros((100, 100), dtype=np.uint8)
        box = mask_to_box(mask)
        assert box is None

    def test_infer_supercategory_damage(self):
        from car_damage_segmentation.data import infer_supercategory
        assert infer_supercategory("dent") == "damage"
        assert infer_supercategory("scratch") == "damage"
        assert infer_supercategory("crack") == "damage"

    def test_infer_supercategory_part(self):
        from car_damage_segmentation.data import infer_supercategory
        assert infer_supercategory("door") == "part"
        assert infer_supercategory("bumper") == "part"

    def test_coco_export(self):
        from car_damage_segmentation.data import export_records_to_coco
        records = [{
            "image_id": 1,
            "image_path": "/fake/path.jpg",
            "width": 100,
            "height": 100,
            "annotations": [{
                "annotation_id": 1,
                "category_id": 1,
                "category_name": "dent",
                "polygon": [25.0, 25.0, 75.0, 25.0, 75.0, 75.0, 25.0, 75.0],
                "holes": [],
            }],
        }]
        categories = [{"id": 1, "name": "dent", "supercategory": "damage"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coco.json"
            export_records_to_coco(records, categories, path)
            assert path.exists()
            coco = json.loads(path.read_text())
            assert "images" in coco
            assert "annotations" in coco
            assert "categories" in coco
            assert len(coco["images"]) == 1
            assert len(coco["annotations"]) == 1


# Metrics tests

class TestMetrics:
    def test_filter_prediction_by_score(self):
        from car_damage_segmentation.metrics import filter_prediction_by_score
        pred = {
            "boxes": torch.tensor([[10., 10., 50., 50.], [60., 60., 90., 90.]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.9, 0.3]),
            "masks": torch.ones((2, 100, 100), dtype=torch.uint8),
        }
        filtered = filter_prediction_by_score(pred, 0.5)
        assert filtered["boxes"].shape[0] == 1
        assert filtered["labels"][0] == 1

    def test_filter_empty_predictions(self):
        from car_damage_segmentation.metrics import filter_prediction_by_score
        pred = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,)),
            "masks": torch.zeros((0, 1, 1), dtype=torch.uint8),
        }
        filtered = filter_prediction_by_score(pred, 0.5)
        assert filtered["boxes"].shape[0] == 0

    def test_pairwise_mask_iou_perfect(self):
        from car_damage_segmentation.metrics import pairwise_mask_iou
        mask = torch.ones((1, 10, 10), dtype=torch.uint8)
        iou = pairwise_mask_iou(mask, mask)
        assert iou.shape == (1, 1)
        assert abs(iou[0, 0].item() - 1.0) < 1e-5

    def test_pairwise_mask_iou_disjoint(self):
        from car_damage_segmentation.metrics import pairwise_mask_iou
        m1 = torch.zeros((1, 10, 10), dtype=torch.uint8)
        m2 = torch.zeros((1, 10, 10), dtype=torch.uint8)
        m1[0, :5, :] = 1
        m2[0, 5:, :] = 1
        iou = pairwise_mask_iou(m1, m2)
        assert abs(iou[0, 0].item() - 0.0) < 1e-5

    def test_metric_accumulator(self):
        from car_damage_segmentation.metrics import MetricAccumulator
        acc = MetricAccumulator(
            true_positives=5,
            false_positives=2,
            false_negatives=3,
            matched_mask_iou_sum=3.5,
            gt_instance_count=8,
        )
        d = acc.to_dict()
        assert d["precision"] == pytest.approx(5 / 7, abs=0.01)
        assert d["recall"] == pytest.approx(5 / 8, abs=0.01)
        assert d["mean_iou"] == pytest.approx(3.5 / 8, abs=0.01)

    def test_metric_accumulator_empty(self):
        from car_damage_segmentation.metrics import MetricAccumulator
        acc = MetricAccumulator()
        d = acc.to_dict()
        assert d["precision"] == 0.0
        assert d["recall"] == 0.0
        assert d["mean_iou"] == 0.0

    def test_greedy_class_aware_match(self):
        from car_damage_segmentation.metrics import greedy_class_aware_match
        iou = torch.tensor([[0.8, 0.1], [0.2, 0.9]])
        pred_labels = torch.tensor([1, 2])
        gt_labels = torch.tensor([1, 2])
        matches = greedy_class_aware_match(iou, pred_labels, gt_labels, 0.5)
        assert len(matches) == 2
        # Первый prediction → первый GT (0.8), второй → второй GT (0.9)
        matched = {(p, g) for _, p, g in matches}
        assert (0, 0) in matched
        assert (1, 1) in matched


# Modeling tests

class TestModeling:
    def test_get_model(self):
        from car_damage_segmentation.modeling import get_instance_segmentation_model
        model = get_instance_segmentation_model(num_classes=5)
        assert model is not None
        # Проверяем forward pass с изображением подходящего размера
        model.eval()
        with torch.no_grad():
            # Модель ожидает изображение [C, H, W] в диапазоне [0, 1]
            dummy = torch.rand(3, 512, 512)  # rand даёт [0, 1], подходящий размер
            output = model([dummy])
            assert isinstance(output, list)
            assert len(output) == 1

    def test_count_parameters(self):
        from car_damage_segmentation.modeling import get_instance_segmentation_model
        model = get_instance_segmentation_model(num_classes=5)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 1_000_000  # Mask R-CNN должен иметь миллионы параметров

    def test_save_load_checkpoint(self):
        from car_damage_segmentation.modeling import (
            get_instance_segmentation_model,
            load_model_from_checkpoint,
        )
        model = get_instance_segmentation_model(num_classes=5)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "test_model.pth"
            torch.save({
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "class_names": ["door", "dent", "scratch", "bumper"],
                "categories": [
                    {"id": 1, "name": "door", "supercategory": "part"},
                    {"id": 2, "name": "dent", "supercategory": "damage"},
                    {"id": 3, "name": "scratch", "supercategory": "damage"},
                    {"id": 4, "name": "bumper", "supercategory": "part"},
                ],
                "history": [],
            }, ckpt_path)

            loaded_model, class_names, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu")
            assert len(class_names) == 4
            assert class_names == ["door", "dent", "scratch", "bumper"]
            assert ckpt["epoch"] == 1


# Baselines tests

class TestBaselines:
    def test_hsv_detector_creation(self):
        from car_damage_segmentation.baselines import HSVDamageDetector
        detector = HSVDamageDetector()
        assert detector.min_contour_area == 200
        assert len(detector.damage_hsv_ranges) >= 3

    def test_hsv_detector_detect(self):
        from car_damage_segmentation.baselines import HSVDamageDetector
        detector = HSVDamageDetector(min_contour_area=50)
        # Создаём изображение с "повреждением": тёмное пятно на светлом фоне
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200  # светло-серый фон
        img[50:100, 60:140] = [30, 30, 30]  # тёмное пятно
        detections = detector.detect(img)
        # Должен найти хотя бы одно тёмное пятно
        assert len(detections) >= 1

    def test_hsv_detector_clean_image(self):
        from car_damage_segmentation.baselines import HSVDamageDetector
        detector = HSVDamageDetector(min_contour_area=100)
        # Зелёное изображение: H~120, S=255, V=255 — не попадает ни в один damage-диапазон
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, 1] = 255
        detections = detector.detect(img)
        assert len(detections) == 0

    def test_get_lightweight_model(self):
        from car_damage_segmentation.baselines import get_lightweight_maskrcnn
        model = get_lightweight_maskrcnn(num_classes=5)
        assert model is not None
        model.eval()
        with torch.no_grad():
            output = model([torch.rand(3, 512, 512)])
            assert isinstance(output, list)

    def test_measure_inference_time(self):
        from car_damage_segmentation.baselines import (
            get_lightweight_maskrcnn,
            measure_inference_time,
        )
        model = get_lightweight_maskrcnn(num_classes=5)
        device = torch.device("cpu")
        model.to(device)
        t = measure_inference_time(model, device)
        assert t > 0
        assert t < 60_000


# Inference tests

class TestInference:
    def test_to_rgb_numpy_from_array(self):
        from car_damage_segmentation.inference import _to_rgb_numpy
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = _to_rgb_numpy(img)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_to_rgb_numpy_from_pil(self):
        from car_damage_segmentation.inference import _to_rgb_numpy
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
        result = _to_rgb_numpy(img)
        assert result.shape == (50, 50, 3)

    def test_to_rgb_numpy_from_tensor(self):
        from car_damage_segmentation.inference import _to_rgb_numpy
        t = torch.rand(3, 64, 64)
        result = _to_rgb_numpy(t)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_summarize_predictions(self):
        from car_damage_segmentation.inference import summarize_predictions
        pred = {
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.95, 0.85]),
            "masks": torch.ones((2, 100, 100), dtype=torch.uint8),
        }
        class_names = ["door", "dent", "scratch"]
        summary = summarize_predictions(pred, class_names)
        assert len(summary) == 2
        assert summary[0]["Класс"] == "door"
        assert summary[0]["Уверенность"] == 0.95
        assert summary[0]["Площадь маски, px"] == 10000
