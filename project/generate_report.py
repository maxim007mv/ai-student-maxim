import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def plot_history(history_path: str, output_dir: str = "outputs"):
    history = []
    
    # Пытаемся найти историю в разных источниках (от самого свежего к старому)
    sources = [
        Path(history_path),
        Path(output_dir) / "last_model.pth",
        Path(output_dir) / "best_model.pth"
    ]
    
    for source in sources:
        if not source.exists():
            continue
            
        if source.suffix == ".json":
            with open(source, 'r') as f:
                history = json.load(f)
        else:
            print(f"Извлекаю историю из чекпоинта: {source.name}")
            ckpt = torch.load(source, map_location='cpu')
            history = ckpt.get("history", [])
            
        if history:
            print(f"Найдена история: {len(history)} эпох(и)")
            break
    
    if not history:
        print("История обучения не найдена!")
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    m_iou = [h['mean_iou'] for h in history]
    precision = [h['precision'] for h in history]
    recall = [h['recall'] for h in history]
    
    plt.figure(figsize=(15, 10))
    
    # Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'o-', label='Train Loss')
    plt.plot(epochs, val_loss, 'o-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # mIoU Plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, m_iou, 'o-', label='mean IoU', color='green')
    plt.title('Mean IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    
    # Precision/Recall Plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs, precision, 'o-', label='Precision', color='blue')
    plt.plot(epochs, recall, 'o-', label='Recall', color='orange')
    plt.title('Precision & Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_report.png")
    print(f"Отчет сохранен в {Path(output_dir) / 'training_report.png'}")

if __name__ == "__main__":
    plot_history("outputs/history.json")
