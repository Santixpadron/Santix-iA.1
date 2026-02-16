"""
Fase 1: Ingeniería de Datos y Preparación del Dataset (TrashNet)
- Resolución: 64x64 píxeles (trade-off latencia vs. grano fino)
- Normalización: [0, 1] vía /255
- Split: 75% entrenamiento / 25% validación
"""

import tensorflow as tf
import os


def prepare_datasets(data_dir, img_size=(64, 64), batch_size=32, val_split=0.25, seed=123):
    """
    Carga y preprocesa el dataset TrashNet desde un directorio con estructura
    de subcarpetas por clase (cardboard, glass, metal, paper, plastic, trash).

    Args:
        data_dir: Ruta al directorio raíz del dataset.
        img_size: Tupla (alto, ancho) para redimensionamiento. Default 64x64.
        batch_size: Tamaño del lote. Default 32 (potencia de 2).
        val_split: Fracción para validación. Default 0.25 (75/25).
        seed: Semilla para reproducibilidad.

    Returns:
        train_ds: tf.data.Dataset de entrenamiento normalizado.
        val_ds: tf.data.Dataset de validación normalizado.
        class_names: Lista de nombres de clase inferidos.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=val_split,
        subset='training',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=val_split,
        subset='validation',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    # Normalización: escalar píxeles al rango [0, 1]
    # Imperativo para estabilidad del gradiente y evitar saturación de activaciones
    AUTOTUNE = tf.data.AUTOTUNE
    normalization = tf.keras.layers.Rescaling(1. / 255)

    train_ds = train_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


if __name__ == '__main__':
    base = os.path.join(os.getcwd(), 'data', 'dataset-resized')
    print('Buscando datos en', base)
    if os.path.exists(base):
        t, v, names = prepare_datasets(base)
        print('Clases:', names)
        print(f'Batches entrenamiento: {tf.data.experimental.cardinality(t).numpy()}')
        print(f'Batches validación: {tf.data.experimental.cardinality(v).numpy()}')
    else:
        print('No hay datos en', base)
