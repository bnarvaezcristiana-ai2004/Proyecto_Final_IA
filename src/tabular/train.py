"""
src/tabular/train.py
====================
Ciclo de entrenamiento, evaluación y guardado de resultados del MLP tabular.

Decisiones de diseño (ISLP Cap. 10):
- Adam como optimizador (lr=1e-3): ISLP 10.7.2 discute SGD estocástico con
  momentum como variante estándar. Adam es su extensión adaptativa más común
  en la práctica (mismo libro, labs PyTorch).
- weight_decay=1e-4: equivalente a regularización L2 (ridge) sobre pesos.
  ISLP 10.7.2 denomina esto "ridge regularization" para redes neuronales:
  "Σ_l ||W_l||²_F" añadida a la pérdida.
- ReduceLROnPlateau: reduce el learning rate cuando la pérdida de validación
  no mejora, estrategia de "network tuning" mencionada en ISLP 10.7.4.
- Guardado del mejor modelo (early stopping por val_loss): ISLP 10.8 señala
  que detener el entrenamiento antes de interpolación completa puede actuar
  como regularización efectiva.
- Métricas: Accuracy, Precision, Recall, F1-Score (ISLP 4.7 sobre métricas
  de clasificación; el libro usa estas métricas en sus evaluaciones).
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")         # sin display de pantalla (compatible con servidor)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ── Rutas de resultados ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "..", "results", "tabular")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Semilla y dispositivo ─────────────────────────────────────────────────────
SEED = 42
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
) -> float:
    """Ejecuta una época de entrenamiento. Retorna la pérdida media."""
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Calcula la pérdida media en un DataLoader sin actualizar pesos."""
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
) -> tuple[list, list, nn.Module]:
    """
    Entrena el MLP con early stopping basado en pérdida de validación.

    ISLP 10.7.2: weight_decay implementa regularización L2 (ridge) sobre
    los pesos de la red, penalizando ||W||² en la función de pérdida.

    ISLP 10.7.4 ("Network Tuning"): ajustar lr y paciencia es clave;
    ReduceLROnPlateau reduce lr cuando el progreso se estanca.

    Parámetros
    ----------
    model          : instancia de MLP ya en DEVICE.
    train_loader   : DataLoader de entrenamiento.
    test_loader    : DataLoader de test (usado como validación aquí).
    num_epochs     : número máximo de épocas.
    lr             : tasa de aprendizaje inicial.
    weight_decay   : coeficiente de regularización L2.
    patience       : épocas sin mejora antes de detener el entrenamiento.

    Retorna
    -------
    (train_losses, val_losses, best_model_state)
    """
    model = model.to(DEVICE)

    # CrossEntropyLoss: aplica log-softmax + NLLLoss (ISLP 10.9.2)
    criterion = nn.CrossEntropyLoss()

    # Adam: extensión adaptativa de SGD con momentum (ISLP 10.7.2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Reduce lr en plateau → "network tuning" (ISLP 10.7.4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"\n[train] Entrenando en {DEVICE} por hasta {num_epochs} épocas ...")
    print(f"[train] Paciencia early-stopping: {patience} épocas\n")
    print(f"{'Época':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print("─" * 46)

    for epoch in range(1, num_epochs + 1):
        tr_loss  = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_loss(model, test_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>12.5f}  {val_loss:>10.5f}  {current_lr:>10.2e}")

        # ── Early stopping (ISLP 10.8) ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n[train] Early stopping en época {epoch} "
                  f"(sin mejora por {patience} épocas).")
            break

    print(f"\n[train] Mejor val_loss: {best_val_loss:.5f}")
    model.load_state_dict(best_weights)   # restaurar mejor modelo
    return train_losses, val_losses, model


# ─────────────────────────────────────────────────────────────────────────────
# CURVA DE PÉRDIDA
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses: list, val_losses: list) -> str:
    """
    Genera y guarda la curva de pérdida (Train vs. Validation).
    Retorna la ruta del archivo guardado.

    La forma de la curva ilustra el bias-variance tradeoff de ISLP Cap. 2:
    si train_loss ↓↓ pero val_loss ↑, hay overfitting.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss",      linewidth=2)
    ax.plot(epochs, val_losses,   label="Validation Loss", linewidth=2, linestyle="--")

    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("CrossEntropy Loss", fontsize=12)
    ax.set_title("Curva de Pérdida — MLP Tabular\n(ISLP Cap. 10)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "mlp_loss_curve.png")
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
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(DEVICE))
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_true)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    label_encoder,
) -> dict:
    """
    Calcula Accuracy, Precision, Recall y F1-Score en el conjunto de test.

    ISLP Cap. 4.7 introduce estas métricas en el contexto de clasificación.
    El promedio 'weighted' es adecuado cuando las clases están desbalanceadas.

    Retorna un diccionario con las métricas y el reporte completo.
    """
    preds, true = predict(model, test_loader)

    avg = "binary" if len(label_encoder.classes_) == 2 else "weighted"

    acc  = accuracy_score(true, preds)
    prec = precision_score(true, preds, average=avg, zero_division=0)
    rec  = recall_score(true, preds,    average=avg, zero_division=0)
    f1   = f1_score(true, preds,         average=avg, zero_division=0)

    class_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(true, preds, target_names=class_names, zero_division=0)

    print("\n" + "═" * 50)
    print("  MÉTRICAS EN TEST — MLP Tabular")
    print("═" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (avg={avg})")
    print(f"  Recall   : {rec:.4f}   (avg={avg})")
    print(f"  F1-Score : {f1:.4f}   (avg={avg})")
    print("\n── Reporte por clase ──")
    print(report)

    # Guardar métricas en texto
    metrics_path = os.path.join(RESULTS_DIR, "mlp_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("MÉTRICAS EN TEST — MLP Tabular\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}  (avg={avg})\n")
        f.write(f"Recall   : {rec:.4f}   (avg={avg})\n")
        f.write(f"F1-Score : {f1:.4f}   (avg={avg})\n\n")
        f.write("Reporte por clase:\n")
        f.write(report)
    print(f"[train] Métricas guardadas → {metrics_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA (ejecutar directamente este archivo)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(_HERE, "..", ".."))

    from src.tabular.data_loader import load_arff, identify_features_target, preprocess
    from src.tabular.model import build_mlp

    # 1. Datos
    df = load_arff()
    X_raw, y_raw, _ = identify_features_target(df)
    train_loader, test_loader, scaler, le, num_classes, in_features = preprocess(
        X_raw, y_raw
    )

    # 2. Modelo
    model = build_mlp(in_features=in_features, num_classes=num_classes)

    # 3. Entrenamiento
    train_losses, val_losses, model = train(
        model, train_loader, test_loader,
        num_epochs=150, lr=1e-3, weight_decay=1e-4, patience=20
    )

    # 4. Curva de pérdida
    plot_loss_curve(train_losses, val_losses)

    # 5. Evaluación
    evaluate(model, test_loader, le)

    # 6. Guardar modelo
    model_path = os.path.join(RESULTS_DIR, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[train] Modelo guardado → {model_path}")