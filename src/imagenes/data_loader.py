"""
src/imagenes/data_loader.py
===========================
Carga el dataset potato_subset con torchvision.datasets.ImageFolder,
aplica transformaciones y divide en 80% train / 20% test.

Decisiones de diseño (ISLP Cap. 10):
- ImageFolder: ISLP 10.9.3 usa esta clase; infiere clases desde subdirectorios.
- Resize(128) → ToTensor → Normalize: transformaciones estándar (ISLP 10.9.3).
- Normalize con medias de ImageNet: práctica estándar para imágenes naturales.
- random_split (80/20): ISLP 10.9 usa random_split para separar conjuntos.
- Data Augmentation solo en train: ISLP 10.3.4 lo presenta como regularización.
  Patrón correcto: dos instancias de ImageFolder con diferente transform,
  mismos índices del split -> augmentation solo en entrenamiento.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Rutas
_HERE = os.path.dirname(os.path.abspath(__file__))
# Apunta directamente a potato_subset (contiene subcarpetas por clase)
DEFAULT_DATA_DIR = os.path.join(
    _HERE, "..", "..", "data", "proyecto_final", "imagenes", "potato_subset"
)
RESULTS_DIR = os.path.join(_HERE, "..", "..", "results", "imagenes")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE   = 128
BATCH_SIZE = 32
SEED       = 42
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]


def get_transforms():
    """Transform con augmentation para train; determinista para test."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return train_transform, test_transform


def load_dataset(data_dir: str = DEFAULT_DATA_DIR):
    """
    Carga con ImageFolder y split 80/20.

    Patrón correcto: dos instancias del mismo directorio con diferente
    transform + mismos indices aleatorios. Asi train recibe augmentation
    y test solo resize+normalize.

    Retorna: train_loader, test_loader, class_names, num_classes
    """
    print(f"[data_loader] Cargando imagenes desde: {data_dir}")

    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(root=data_dir, transform=test_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    n_total     = len(train_dataset)

    print(f"[data_loader] Clases ({num_classes}): {class_names}")
    print(f"[data_loader] Total imagenes: {n_total}")

    # Split 80/20 con indices aleatorios fijos
    n_train = int(0.80 * n_total)
    n_test  = n_total - n_train

    generator     = torch.Generator().manual_seed(SEED)
    indices       = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    test_indices  = indices[n_train:]

    train_subset = Subset(train_dataset, train_indices)
    test_subset  = Subset(test_dataset,  test_indices)

    print(f"[data_loader] Train: {n_train}  |  Test: {n_test}")

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, class_names, num_classes