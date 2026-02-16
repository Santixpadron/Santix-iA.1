"""
Fase 2: Arquitectura CNN + Fase 3: Pipeline de Entrenamiento

Arquitectura CNN ligera para clasificación de residuos (TrashNet):
- 3 Bloques Convolucionales: Conv2D(32, 2x2, same, ReLU) → MaxPool(2x2) → Dropout(0.2)
- Flatten → Dense(512, ReLU) → Dense(num_classes, Softmax)
- Capa GaussianNoise como suavizado de entrada (defensa adversaria FocalX)

Pipeline de entrenamiento:
- Optimizador: Adam (tasa adaptativa)
- Pérdida: Categorical Cross-Entropy (multiclase)
- Épocas: 70, Batch: 32
- EarlyStopping: monitor val_loss, patience 10
- ModelCheckpoint: guardar mejor modelo por val_loss
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import prepare_datasets


def build_model(input_shape=(64, 64, 3), num_classes=6):
    """
    Construye la CNN personalizada según la especificación técnica.

    Arquitectura:
        - GaussianNoise(0.01): Suavizado de entrada / defensa adversaria (FocalX)
        - 3x [Conv2D(32, 2x2, same, ReLU) → MaxPooling2D(2x2) → Dropout(0.2)]
        - Flatten → Dense(512, ReLU) → Dense(num_classes, Softmax)

    Args:
        input_shape: Forma de entrada (alto, ancho, canales). Default (64, 64, 3).
        num_classes: Número de clases de salida. Default 6 (TrashNet).

    Returns:
        modelo Keras compilado.
    """
    model = models.Sequential([
        # Capa de entrada
        layers.Input(shape=input_shape),

        # Capa de validación/suavizado de entrada (seguridad FocalX)
        # Mitiga ataques adversarios tipo FGSM añadiendo ruido gaussiano
        layers.GaussianNoise(0.01),

        # --- Bloque Convolucional 1 ---
        layers.Conv2D(32, (2, 2), padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # --- Bloque Convolucional 2 ---
        layers.Conv2D(32, (2, 2), padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # --- Bloque Convolucional 3 ---
        layers.Conv2D(32, (2, 2), padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # --- Capas Densas ---
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])

    return model


def main():
    # Ruta del dataset configurable por variable de entorno
    data_dir = os.environ.get(
        'DATA_DIR',
        os.path.join(os.getcwd(), 'detector_residuos', 'data', 'dataset-resized')
    )
    os.makedirs('models', exist_ok=True)

    # Fase 1: Carga y preprocesamiento
    train_ds, val_ds, class_names = prepare_datasets(data_dir)
    print(f'Clases detectadas ({len(class_names)}): {class_names}')

    # Fase 2: Construcción del modelo
    model = build_model(num_classes=len(class_names))

    # Fase 3: Compilación con Adam y Categorical Cross-Entropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks de control
    callbacks = [
        # EarlyStopping: detener si val_loss no mejora en 10 épocas
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # ModelCheckpoint: persistir la mejor versión del modelo
        tf.keras.callbacks.ModelCheckpoint(
            'models/model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
    ]

    # Entrenamiento: 70 épocas, batch 32 (definido en preprocess)
    epochs = int(os.environ.get('EPOCHS', '70'))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Guardar metadatos de clases
    with open('models/class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

    # Guardar historial de entrenamiento
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open('models/training_history.json', 'w', encoding='utf-8') as f:
        json.dump(hist_dict, f, ensure_ascii=False, indent=2)

    print(f'\nEntrenamiento completado. Modelo guardado en models/model.h5 (épocas={epochs})')
    print(f'Mejor val_loss: {min(history.history["val_loss"]):.4f}')
    print(f'Mejor val_accuracy: {max(history.history["val_accuracy"]):.4f}')


if __name__ == '__main__':
    main()
