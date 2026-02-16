"""
Fase 1 + Optimización: Ingeniería de Datos con Data Augmentation

- Resolución: 128x128 píxeles (mejora sobre 64x64 para capturar más detalle)
- Normalización: [0, 1] vía /255
- Split: 75% entrenamiento / 25% validación
- Data Augmentation: rotación, flip, zoom, brillo (solo en entrenamiento)
"""

import tensorflow as tf
import os


def create_augmentation_layer():
    """
    Crea un pipeline de Data Augmentation para reducir overfitting.
    Solo se aplica durante entrenamiento (no en inferencia).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name='data_augmentation')


def prepare_datasets(data_dir, img_size=(128, 128), batch_size=32, val_split=0.25, seed=123, augment=True):
    """
    Carga y preprocesa el dataset TrashNet con Data Augmentation opcional.

    Args:
        data_dir: Ruta al directorio raíz del dataset.
        img_size: Tupla (alto, ancho) para redimensionamiento. Default 128x128.
        batch_size: Tamaño del lote. Default 32.
        val_split: Fracción para validación. Default 0.25 (75/25).
        seed: Semilla para reproducibilidad.
        augment: Si True, aplica Data Augmentation al set de entrenamiento.

    Returns:
        train_ds: tf.data.Dataset de entrenamiento normalizado (con augmentation).
        val_ds: tf.data.Dataset de validación normalizado (sin augmentation).
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

    AUTOTUNE = tf.data.AUTOTUNE
    normalization = tf.keras.layers.Rescaling(1. / 255)

    if augment:
        augmentation = create_augmentation_layer()
        # Entrenamiento: augmentation + normalización
        train_ds = train_ds.map(
            lambda x, y: (normalization(augmentation(x, training=True)), y)
        ).cache().prefetch(buffer_size=AUTOTUNE)
    else:
        train_ds = train_ds.map(
            lambda x, y: (normalization(x), y)
        ).cache().prefetch(buffer_size=AUTOTUNE)

    # Validación: solo normalización (sin augmentation)
    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y)
    ).cache().prefetch(buffer_size=AUTOTUNE)

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
