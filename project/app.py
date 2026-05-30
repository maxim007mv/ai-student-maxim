from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from car_damage_segmentation.inference import predict_and_visualize
from car_damage_segmentation.modeling import load_model_from_checkpoint
from car_damage_segmentation.utils import resolve_device


st.set_page_config(
    page_title="Car Damage Vision",
    page_icon="🚗",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 15% 20%, rgba(255, 172, 94, 0.20), transparent 24%),
                radial-gradient(circle at 85% 10%, rgba(71, 173, 255, 0.16), transparent 24%),
                linear-gradient(145deg, #f6f1e8 0%, #f0f5fb 50%, #eef5f0 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            padding: 28px 32px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(18, 44, 62, 0.95), rgba(6, 78, 59, 0.88));
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
            color: #f8fafc;
            position: relative;
            overflow: hidden;
            margin-bottom: 22px;
        }

        .hero:before {
            content: "";
            position: absolute;
            width: 360px;
            height: 360px;
            right: -120px;
            top: -180px;
            background: radial-gradient(circle, rgba(251, 191, 36, 0.42), transparent 60%);
        }

        .hero h1 {
            font-size: 2.2rem;
            margin: 0 0 10px 0;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0;
            font-size: 1.02rem;
            max-width: 860px;
            color: rgba(248, 250, 252, 0.9);
            line-height: 1.6;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.65);
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
            border-radius: 24px;
            padding: 18px 20px;
            backdrop-filter: blur(10px);
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 8px 0 4px 0;
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(244,248,252,0.88));
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid rgba(203, 213, 225, 0.65);
        }

        .metric-card .label {
            font-size: 0.88rem;
            color: #475569;
            margin-bottom: 6px;
        }

        .metric-card .value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_cached_model(checkpoint_path: str, device_name: str):
    return load_model_from_checkpoint(checkpoint_path, device=device_name)


def render_metrics(detections: list[dict]) -> None:
    num_detections = len(detections)
    avg_confidence = sum(item["Уверенность"] for item in detections) / num_detections if num_detections else 0.0
    unique_classes = len({item["Класс"] for item in detections})

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">Найдено экземпляров</div>
                <div class="value">{num_detections}</div>
            </div>
            <div class="metric-card">
                <div class="label">Средняя уверенность</div>
                <div class="value">{avg_confidence:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Уникальных классов</div>
                <div class="value">{unique_classes}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

st.markdown(
    """
    <div class="hero">
        <h1>Car Damage Vision</h1>
        <p>
            Загрузите фотографию автомобиля, и приложение автоматически найдёт повреждения и детали,
            наложит полупрозрачные маски, bounding boxes и подпишет классы с confidence score.
            Подходит как демонстрационный интерфейс для академического проекта по instance segmentation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Параметры инференса")
    checkpoint_path = st.text_input("Путь к чекпоинту", value="outputs/best_model.pth")
    score_threshold = st.slider("Порог уверенности", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    alpha = st.slider("Прозрачность масок", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
    device_name = str(resolve_device())
    st.caption(f"Устройство: `{device_name}`")
    st.caption("Запуск приложения: `streamlit run app.py`")

if not Path(checkpoint_path).exists():
    st.warning("Чекпоинт не найден. Сначала обучите модель командой `python train_instance_segmentation.py`.")
    st.stop()

uploaded_file = st.file_uploader(
    "Перетащите фотографию автомобиля",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Загружаю модель и запускаю инференс..."):
        model, class_names, _ = load_cached_model(checkpoint_path, device_name)
        result = predict_and_visualize(
            image_tensor=image,
            model=model,
            class_names=class_names,
            score_threshold=score_threshold,
            alpha=alpha,
            device=device_name,
            show=False,
            return_data=True,
        )

    detections = result["detections"]
    render_metrics(detections)

    left_column, right_column = st.columns([1.0, 1.08], gap="large")
    with left_column:
        st.markdown('<div class="section-title">Исходное изображение</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with right_column:
        st.markdown('<div class="section-title">Результат сегментации</div>', unsafe_allow_html=True)
        st.image(result["rendered_image"], use_container_width=True)

    st.markdown('<div class="section-title">Таблица найденных объектов</div>', unsafe_allow_html=True)
    if detections:
        detections_frame = pd.DataFrame(detections)
        st.dataframe(detections_frame, use_container_width=True, hide_index=True)
    else:
        st.info("На текущем пороге уверенности модель не нашла объектов.")
