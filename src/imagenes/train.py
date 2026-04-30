"""
src/imagenes/train.py
=====================
Ciclo de entrenamiento, evaluación y guardado de resultados de la CNN.

Decisiones de diseño (ISLP Cap. 10):
- Adam + weight_decay (ridge): ISLP 10.7.2 presenta ambas regularizaciones
  (ridge y dropout) como complementarias para evitar overfitting en CNNs.
- ReduceLROnPlateau: ISLP 10.7.4 ("Network Tuning") recomienda ajustar lr
  dinámicamente cuando el modelo deja de mejorar.
- Early stopping (ISLP 10.8): "early stopping during SGD can serve as a
  form of regularization that prevents us from interpolating the training
  data, while still getting very good results on test data."
- Matriz de confusión: ISLP 4.5.1 introduce esta herramienta para evaluar
  clasificadores multiclase, visualizando errores por clase.
- Accuracy por clase (diagonal de la CM normalizada): extensión natural
  de la discusión de ISLP 4.7 sobre métricas de clasificación.
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Rutas ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "..", "results", "imagenes")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dispositivo ───────────────────────────────────────────────────────────────
SEED   = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Ejecuta una época de entrenamiento. Retorna (loss_media, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Calcula pérdida y accuracy en un DataLoader. Sin actualizar pesos."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits        = model(imgs)
        running_loss += criterion(logits, labels).item() * imgs.size(0)
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> tuple[dict, nn.Module]:
    """
    Entrena la CNN con early stopping.

    ISLP 10.7.2: weight_decay → regularización L2 (ridge) sobre los pesos.
    ISLP 10.7.3: Dropout aplicado dentro del modelo (model.py).
    ISLP 10.8  : Early stopping como regularización implícita.

    Retorna
    -------
    (history_dict, best_model)
    history_dict contiene: train_loss, val_loss, train_acc, val_acc
    """
    model = model.to(DEVICE)

    # CrossEntropyLoss: log-softmax + NLLLoss (ISLP 10.9.2–10.9.3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss  = float("inf")
    best_weights   = copy.deepcopy(model.state_dict())
    epochs_no_imp  = 0

    print(f"\n[train] Entrenando CNN en {DEVICE} por hasta {num_epochs} épocas ...")
    print(f"[train] Paciencia early-stopping: {patience}\n")
    header = f"{'Época':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'TrainAcc':>10}  {'ValAcc':>10}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate_epoch(model, test_loader, criterion)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>10.5f}  {vl_loss:>10.5f}  "
                  f"{tr_acc:>10.4f}  {vl_acc:>10.4f}")

        # ── Early stopping (ISLP 10.8) ────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights  = copy.deepcopy(model.state_dict())
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
        if epochs_no_imp >= patience:
            print(f"\n[train] Early stopping en época {epoch}.")
            break

    model.load_state_dict(best_weights)
    print(f"\n[train] Mejor val_loss: {best_val_loss:.5f}")
    return history, model


# ─────────────────────────────────────────────────────────────────────────────
# CURVA DE PÉRDIDA
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history: dict) -> str:
    """
    Genera y guarda la gráfica de pérdida y accuracy por época.
    Retorna la ruta del archivo guardado.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # ── Pérdida ──────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], label="Train Loss",      linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Validation Loss", linewidth=2,
             linestyle="--")
    ax1.set_xlabel("Época", fontsize=12)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=12)
    ax1.set_title("Curva de Pérdida — CNN\n(ISLP Cap. 10)", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────────────
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=2,
             color="tab:green")
    ax2.plot(epochs, history["val_acc"],   label="Val Accuracy",   linewidth=2,
             linestyle="--", color="tab:orange")
    ax2.set_xlabel("Época", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy por Época — CNN", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "cnn_loss_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[train] Curva de pérdida guardada → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Genera predicciones y etiquetas verdaderas en modo eval."""
    model.eval()
    all_preds, all_true = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(DEVICE))
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())
    return np.array(all_preds), np.array(all_true)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """
    Genera y guarda la matriz de confusión con accuracy por clase.

    ISLP 4.5.1 introduce la matriz de confusión para evaluar clasificadores.
    La diagonal normalizada (accuracy por clase) indica qué clases son
    difíciles de distinguir para el modelo.

    Retorna la ruta del archivo guardado.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # accuracy por clase

    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 2 + 2), max(5, n)))

    # ── Matriz absoluta ───────────────────────────────────────────────────────
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_xlabel("Predicción", fontsize=11)
    axes[0].set_ylabel("Real", fontsize=11)
    axes[0].set_title("Matriz de Confusión\n(conteos)", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    # ── Matriz normalizada (accuracy por clase) ───────────────────────────────
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
    )
    axes[1].set_xlabel("Predicción", fontsize=11)
    axes[1].set_ylabel("Real", fontsize=11)
    axes[1].set_title("Matriz de Confusión\n(accuracy por clase)", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[train] Matriz de confusión guardada → {out}")

    # Imprimir accuracy por clase en consola
    print("\n── Accuracy por clase ──────────────────")
    for i, cls in enumerate(class_names):
        print(f"  {cls:<25}: {cm_norm[i, i]:.4f}")
    return out


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
) -> dict:
    """
    Calcula Accuracy global, genera el reporte por clase y la matriz de
    confusión. Guarda resultados en RESULTS_DIR.

    Retorna diccionario con accuracy y ruta de la matriz de confusión.
    """
    preds, true = predict(model, test_loader)

    acc    = accuracy_score(true, preds)
    report = classification_report(true, preds, target_names=class_names,
                                   zero_division=0)

    print("\n" + "═" * 52)
    print("  MÉTRICAS EN TEST — CNN Imágenes")
    print("═" * 52)
    print(f"  Accuracy global: {acc:.4f}")
    print("\n── Reporte por clase ──")
    print(report)

    # Guardar métricas en texto
    metrics_path = os.path.join(RESULTS_DIR, "cnn_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("MÉTRICAS EN TEST — CNN Imágenes\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy global: {acc:.4f}\n\n")
        f.write("Reporte por clase:\n")
        f.write(report)
    print(f"[train] Métricas guardadas → {metrics_path}")

    # Matriz de confusión
    cm_path = plot_confusion_matrix(true, preds, class_names)

    return {"accuracy": acc, "cm_path": cm_path}


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(_HERE, "..", ".."))

    from src.imagenes.data_loader import load_dataset
    from src.imagenes.model import build_cnn

    # 1. Datos
    train_loader, test_loader, class_names, num_classes = load_dataset()

    # 2. Modelo
    model = build_cnn(num_classes=num_classes)

    # 3. Entrenamiento
    history, model = train(
        model, train_loader, test_loader,
        num_epochs=60, lr=1e-3, weight_decay=1e-4, patience=12
    )

    # 4. Curva de pérdida
    plot_loss_curve(history)

    # 5. Evaluación y matriz de confusión
    evaluate(model, test_loader, class_names)

    # 6. Guardar modelo
    model_path = os.path.join(RESULTS_DIR, "cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[train] Modelo guardado → {model_path}")