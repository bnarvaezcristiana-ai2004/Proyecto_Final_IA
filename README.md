# Proyecto Final — Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)

Implementación de dos modelos de clasificación con redes neuronales profundas usando **PyTorch**, basados en los principios del libro *An Introduction to Statistical Learning with Applications in Python* (ISLP) — James et al., 2023.

---

## Descripción

El proyecto clasifica datos tabulares e imágenes usando redes neuronales entrenadas desde cero, aplicando buenas prácticas de preprocesamiento, regularización y evaluación.

| Componente | Datos | Modelo | Accuracy |
|---|---|---|---|
| **Tabular** | `phpnThNfi.arff` — 4 839 instancias, 2 clases | MLP | **98.86 %** |
| **Imágenes** | `potato_subset` — 456 imágenes RGB, 3 clases | CNN | **94.57 %** |

---

## Estructura
Proyecto_Final_IA/
├── data/
│   └── proyecto_final/
│       ├── tabular/
│       │   └── phpnThNfi.arff
│       └── imagenes/
│           └── potato_subset/
│               ├── Potato___Early_blight/
│               ├── Potato___Late_blight/
│               └── Potato___healthy/
├── src/
│   ├── tabular/
│   │   ├── data_loader.py
│   │   ├── model.py
│   │   └── train.py
│   └── imagenes/
│       ├── data_loader.py
│       ├── model.py
│       └── train.py
├── results/
│   ├── tabular/
│   │   ├── mlp_loss_curve.png
│   │   ├── mlp_metrics.txt
│   │   └── mlp_model.pth
│   └── imagenes/
│       ├── cnn_loss_curve.png
│       ├── cnn_confusion_matrix.png
│       ├── cnn_metrics.txt
│       └── cnn_model.pth
├── latex/
│   ├── main.tex
│   ├── referencias.bib
│   └── main.pdf
├── requirements.txt
└── README.md

## Instalación y Ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/bnarvaezcristiana-ai2004/Proyecto_Final_IA.git
cd Proyecto_Final_IA

# 2. Activar entorno virtual
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
python src/tabular/train.py    # Componente Tabular — MLP
python src/imagenes/train.py   # Componente Imágenes — CNN
```

---

## Arquitecturas

### MLP — Tabular─ README.md

Entrada(5) → Linear(256) → BN → ReLU → Dropout(0.4)
→ Linear(128) → BN → ReLU → Dropout(0.4)
→ Linear(2) [logits]
Parámetros: 35 458 | Early stopping: época 47

### CNN — Imágenes

Entrada(3×128×128)
→ Conv(32)  → BN → ReLU → MaxPool  →  32×64×64
→ Conv(64)  → BN → ReLU → MaxPool  →  64×32×32
→ Conv(128) → BN → ReLU → MaxPool  → 128×16×16
→ Flatten → Linear(512) → Dropout(0.5) → Linear(3)
Parámetros: 16 872 963 | Early stopping: época 28

---

## Resultados

### MLP — Datos Tabulares (Test: 968 instancias)

| Clase | Precision | Recall | F1-Score |
|---|---|---|---|
| Clase 1 — mayoritaria (n=916) | 0.9919 | 0.9934 | 0.9927 |
| Clase 2 — minoritaria (n=52)  | 0.9020 | 0.8846 | 0.8932 |
| **Accuracy global** | | | **0.9886** |

### CNN — Imágenes de Papa (Test: 92 imágenes)

| Clase | Precision | Recall | F1-Score |
|---|---|---|---|
| Early Blight | 0.9375 | 0.9375 | 0.9375 |
| Late Blight  | 0.9000 | 0.9286 | 0.9141 |
| Healthy      | 1.0000 | 0.9688 | 0.9841 |
| **Weighted avg** | **0.9457** | **0.9457** | **0.9453** |

---

## Decisiones de Diseño (basadas en ISLP)

| Técnica | Justificación | Capítulo ISLP |
|---|---|---|
| `StandardScaler` solo sobre train | Evita data leakage | Cap. 10.9.1 |
| Activación ReLU | Mitiga gradiente evanescente | Cap. 10.1 |
| Dropout (0.4 / 0.5) | Regularización estocástica | Cap. 10.7.3 |
| Weight decay L2 | Penaliza norma de pesos | Cap. 10.7.2 |
| Early stopping | Regularización implícita | Cap. 10.8 |
| Data augmentation solo en train | Evita contaminar el test | Cap. 10.3.4 |
| CrossEntropyLoss sin Softmax | Estabilidad numérica en PyTorch | Cap. 10.9.2 |

---

## Referencia

> James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
> *An Introduction to Statistical Learning with Applications in Python* (1st ed.). Springer.
> https://www.statlearning.com

---

## Autor

**Cristian Bravo** — Curso de Inteligencia Artificial, 2026
