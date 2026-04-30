"""
src/imagenes/model.py
=====================
Arquitectura CNN para clasificación de imágenes de papa (potato_subset).

Decisiones de diseño (ISLP Cap. 10):

Conv2d + ReLU + MaxPool2d (ISLP 10.3.1 y 10.3.2):
  "A convolution layer takes as input the image in a certain region, and
   produces a single number as output" (10.3.1). MaxPool reduce la dimensión
   espacial conservando las características más prominentes (10.3.2:
   "reduces the dimension of the feature map").

Tres bloques convolucionales con filtros crecientes (32→64→128):
  ISLP 10.3.3 ("Architecture of a CNN") señala que las CNNs profundas
  aumentan progresivamente el número de filtros a medida que la resolución
  espacial decrece, capturando características cada vez más abstractas.

Softmax / CrossEntropyLoss:
  ISLP 10.3 usa Softmax para clasificación multiclase. PyTorch integra
  log-softmax en nn.CrossEntropyLoss, por lo que la capa de salida
  emite logits sin activación. Esto es numéricamente más estable.
  (PyTorch docs + práctica confirmada en ISLP 10.9.2 y 10.9.3).

Dropout(0.5) en capas FC:
  ISLP 10.7.3: "Dropout learning can be used at each layer". Una tasa de
  0.5 es la recomendación canónica para capas densas (fully-connected).

BatchNorm2d después de cada convolución:
  Complemento natural a la normalización de datos discutida en ISLP 10.9.1;
  acelera la convergencia y actúa como regularizador implícito.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Red Neuronal Convolucional para clasificación de imágenes 128×128 RGB.

    Arquitectura
    ------------
    Bloque Conv 1: Conv2d(3→32, 3×3)  → BatchNorm → ReLU → MaxPool(2×2)  → 64×64
    Bloque Conv 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  → 32×32
    Bloque Conv 3: Conv2d(64→128, 3×3)→ BatchNorm → ReLU → MaxPool(2×2)  → 16×16
    Flatten → Linear(128·16·16 → 512) → ReLU → Dropout(0.5)
            → Linear(512 → K)  [logits para CrossEntropyLoss]

    Parámetros
    ----------
    num_classes  : número de clases K.
    dropout_rate : tasa de dropout en capas FC (ISLP 10.7.3).
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()

        # ── Bloque Convolucional 1 ────────────────────────────────────────────
        # ISLP 10.3.1: "each filter has a small spatial footprint (3×3)"
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),        # ISLP 10.1: ReLU, activación preferida
            nn.MaxPool2d(kernel_size=2),  # ISLP 10.3.2: pooling 2×2
        )

        # ── Bloque Convolucional 2 ────────────────────────────────────────────
        # Filtros × 2 al reducir resolución espacial (ISLP 10.3.3)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # ── Bloque Convolucional 3 ────────────────────────────────────────────
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # Después de 3 MaxPool(2): 128 → 64 → 32 → 16  (por lado)
        # Feature map: 128 canales × 16 × 16 = 32.768 neuronas

        # ── Capas Totalmente Conectadas (FC) ─────────────────────────────────
        # ISLP 10.3.3: "the output feature maps are then flattened and fed
        #  into a fully-connected layer"
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),   # ISLP 10.7.3
            nn.Linear(512, num_classes),  # logits; Softmax implícito en CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante:
          x  : (batch, 3, 128, 128)
          out: (batch, num_classes)  ← logits sin Softmax
        """
        x = self.conv_block1(x)   # → (batch, 32, 64, 64)
        x = self.conv_block2(x)   # → (batch, 64, 32, 32)
        x = self.conv_block3(x)   # → (batch, 128, 16, 16)
        x = self.classifier(x)    # → (batch, K)
        return x


def build_cnn(num_classes: int, dropout_rate: float = 0.5) -> SimpleCNN:
    """
    Función de fábrica para instanciar la CNN.
    Imprime un resumen de la arquitectura con dimensiones de cada capa.
    """
    model = SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    print("\n[model] Arquitectura CNN:")
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parámetros entrenables: {n_params:,}\n")
    return model