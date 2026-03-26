from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Настройка шрифтов для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # Для корректного отображения минуса

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Для числовых колонок строит по отдельной гистограмме.
    Возвращает список путей к PNG.
    """
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    for i, name in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[name].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(s.values, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_title(f"Гистограмма распределения {name}", fontsize=12, fontweight='bold')
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("Частота", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"hist_{i+1}_{name}.png"
        fig.savefig(out_path)
        plt.close(fig)

        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Простая визуализация пропусков: где True=пропуск, False=значение.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Рисуем пустой график
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Пустой датасет", ha="center", va="center")
        ax.axis("off")
    else:
        mask = df.isna().values
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 0.5), max(6, df.shape[0] * 0.02)))
        ax.imshow(mask, aspect="auto", interpolation="none", cmap="binary")
        ax.set_xlabel("Колонки", fontsize=10)
        ax.set_ylabel("Строки", fontsize=10)
        ax.set_title("Матрица пропущенных значений", fontsize=12, fontweight='bold')
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляции числовых признаков.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Недостаточно числовых колонок для корреляции", ha="center", va="center")
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(8, corr.shape[1] * 0.8), max(6, corr.shape[0] * 0.6)))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(corr.shape[0]))
        ax.set_yticklabels(corr.index, fontsize=9)
        ax.set_title("Тепловая карта корреляции", fontsize=12, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Коэффициент корреляции Пирсона", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)
    return paths
