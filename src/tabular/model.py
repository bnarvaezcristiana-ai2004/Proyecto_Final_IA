"""
src/tabular/model.py
====================
Arquitectura MLP para clasificación tabular.

Decisiones de diseño (ISLP Cap. 10):
- Dos capas ocultas (256 → 128): ISLP 10.9.2 muestra que redes con múltiples
  capas ocultas superan a las de una sola capa en MNIST. La regla general es
  comenzar con pocas capas y aumentar si hay evidencia de underfitting.
- ReLU como función de activación (nn.ReLU): ISLP 10.1 destaca que ReLU es
  "the preferred choice in modern neural networks" por su eficiencia
  computacional y por no sufrir el problema del gradiente evanescente.
- Dropout(0.4): ISLP 10.7.3 introduce Dropout Learning como regularización
  estocástica. El libro reporta que con dropout la tasa de error en MNIST
  cae por debajo del 2%. Desactivado automáticamente en model.eval().
- Capa final sin activación: CrossEntropyLoss de PyTorch aplica internamente
  log-softmax + NLLLoss, por lo que la capa de salida entrega logits crudos.
  ISLP 10.9.2 y la documentación de PyTorch confirman esta práctica.
- BatchNorm1d entre Linear y ReLU: estabiliza el entrenamiento normalizando
  las activaciones intermedias (técnica estándar en redes modernas, extensión
  natural de la regularización discutida en ISLP 10.7).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Perceptrón Multicapa para clasificación con K clases.

    Arquitectura
    ------------
    Input(p)
      → Linear(p → 256) → BatchNorm → ReLU → Dropout(0.4)
      → Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)
      → Linear(128 → K)   [logits para CrossEntropyLoss]

    Parámetros
    ----------
    in_features : int
        Número de variables de entrada (columnas de X tras preprocesado).
    num_classes : int
        Número de clases (K).
    dropout_rate : float
        Fracción de neuronas desactivadas aleatoriamente durante el
        entrenamiento (ISLP 10.7.3 usa 0.4 en el ejemplo de MNIST).
    """

    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.4):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            # ── Capa oculta 1 ────────────────────────────────────────────
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),          # estabiliza el entrenamiento
            nn.ReLU(),                    # ISLP 10.1: activación preferida
            nn.Dropout(dropout_rate),     # ISLP 10.7.3: regularización

            # ── Capa oculta 2 ────────────────────────────────────────────
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # ── Capa de salida (logits) ───────────────────────────────────
            # Sin activación: CrossEntropyLoss aplica log-softmax internamente.
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_mlp(in_features: int, num_classes: int, dropout_rate: float = 0.4) -> MLP:
    """
    Función de fábrica para instanciar el MLP.
    Imprime un resumen de la arquitectura.
    """
    model = MLP(in_features=in_features, num_classes=num_classes,
                dropout_rate=dropout_rate)
    print("\n[model] Arquitectura MLP:")
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parámetros entrenables: {n_params:,}\n")
    return model