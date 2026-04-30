"""
src/tabular/data_loader.py
==========================
Carga y preprocesa el archivo phpnThNfi.arff para el componente tabular.

Decisiones de diseño (ISLP Cap. 10):
- StandardScaler: ISLP sección 10.9.1 normaliza explícitamente antes de
  entrenar cualquier red neuronal ("we first normalize the features using a
  StandardScaler() transform"). Las redes neuronales son sensibles a la
  escala de las entradas.
- train_test_split estratificado (80/20): garantiza que la distribución de
  clases se preserve en ambas particiones, práctica estándar del libro.
- LabelEncoder para y: convierte etiquetas string a enteros 0..K-1,
  requeridos por nn.CrossEntropyLoss.
- pd.get_dummies para X categórico: convierte variables nominales en
  representación one-hot antes de pasarlas a la red.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# ── Constante de ruta ────────────────────────────────────────────────────────
# Ajusta esta ruta si la ejecutas desde otro directorio de trabajo.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARFF = os.path.join(
    _HERE, "..", "..", "data", "proyecto_final", "tabular", "phpnThNfi.arff"
)

# ── Semilla global ───────────────────────────────────────────────────────────
SEED = 42


def load_arff(path: str = DEFAULT_ARFF) -> pd.DataFrame:
    """
    Carga un archivo .arff con scipy.io.arff y lo convierte a DataFrame.

    Decodifica automáticamente columnas de tipo bytes (b'valor') a str,
    ya que scipy devuelve strings como bytes en Python 3.
    """
    raw_data, meta = arff.loadarff(path)
    df = pd.DataFrame(raw_data)

    # Decodificar columnas de tipo objeto (bytes → str)
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].str.decode("utf-8")

    print(f"[data_loader] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"[data_loader] Columnas: {df.columns.tolist()}")
    return df


def identify_features_target(df: pd.DataFrame):
    """
    Identifica automáticamente:
    - X: todas las columnas numéricas/categóricas excepto la última.
    - y: la última columna (convención de ARFF: @attribute class {…}).

    Retorna (X_raw, y_raw, target_name).
    """
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1].tolist()
    print(f"[data_loader] Variable objetivo: '{target_col}'")
    print(f"[data_loader] Variables de entrada ({len(feature_cols)}): {feature_cols[:5]} ...")
    return df[feature_cols], df[target_col], target_col


def preprocess(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    test_size: float = 0.20,
    seed: int = SEED,
):
    """
    Preprocesa X e y y devuelve DataLoaders listos para PyTorch.

    Pasos (siguiendo ISLP 10.9.1):
    1. One-hot encoding de columnas categóricas en X.
    2. LabelEncoder en y → enteros 0..K-1.
    3. train_test_split estratificado 80/20.
    4. StandardScaler ajustado solo sobre train (evita data leakage).
    5. Conversión a tensores y empaquetado en DataLoader (batch_size=64).

    Retorna
    -------
    train_loader, test_loader, scaler, label_encoder, num_classes, in_features
    """
    # ── 1. Codificar variables categóricas en X ──────────────────────────────
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    # Convertir bool → float (pd.get_dummies puede producir bool en pandas ≥2.0)
    bool_cols = X_encoded.select_dtypes(include=bool).columns
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(float)
    X = X_encoded.values.astype(np.float32)

    # ── 2. Codificar etiquetas ───────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    num_classes = len(le.classes_)
    print(f"[data_loader] Clases ({num_classes}): {le.classes_}")

    # ── 3. Split 80 / 20 estratificado ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"[data_loader] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 4. Normalización (StandardScaler) ────────────────────────────────────
    # ISLP 10.9.1: "we first normalize the features using StandardScaler()"
    # fit() solo sobre datos de entrenamiento para evitar data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── 5. Tensores y DataLoaders ────────────────────────────────────────────
    t_Xtr = torch.from_numpy(X_train)
    t_ytr = torch.from_numpy(y_train)
    t_Xte = torch.from_numpy(X_test)
    t_yte = torch.from_numpy(y_test)

    # batch_size=64: ISLP recomienda mini-batches pequeños para SGD estocástico
    train_loader = DataLoader(
        TensorDataset(t_Xtr, t_ytr),
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(t_Xte, t_yte),
        batch_size=64,
        shuffle=False,
    )

    in_features = X_train.shape[1]
    return train_loader, test_loader, scaler, le, num_classes, in_features