"""
Utilidades para inferencia y carga del modelo.
Resolución estándar: 64x64 (consistente con el entrenamiento).
"""

import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json
import os


def load_and_preprocess_image_bytes(image_bytes, img_size=(64, 64)):
    """
    Preprocesa una imagen desde bytes para inferencia.

    Args:
        image_bytes: Bytes de la imagen (jpg/png).
        img_size: Tupla (alto, ancho). Default 64x64 para consistencia con entrenamiento.

    Returns:
        Array numpy con shape (1, 64, 64, 3) normalizado a [0, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr.astype('float32'), axis=0)


def load_class_names(models_dir='models'):
    """Carga los nombres de clase desde el JSON generado en entrenamiento."""
    path = os.path.join(models_dir, 'class_names.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_model(models_dir='models'):
    """Carga el modelo entrenado desde disco."""
    model_path = os.path.join(models_dir, 'model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError('No se encontró el modelo. Entrene primero con train.py')
    return tf.keras.models.load_model(model_path)
