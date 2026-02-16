"""
Fase 2+3 Optimizada: CNN con filtros progresivos y BatchNormalization

Arquitectura CNN mejorada:
- 3 Bloques Convolucionales con filtros progresivos (32→64→128)
- BatchNormalization para estabilizar gradientes
- GaussianNoise como defensa adversaria (FocalX)
- Data Augmentation integrada en el pipeline de datos

Pipeline de entrenamiento:
- Optimizador: Adam con ReduceLROnPlateau
- Pérdida: Categorical Cross-Entropy
- Épocas: 70, Batch: 32
- EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import prepare_datasets


def build_model(input_shape=(128, 128, 3), num_classes=6):
    """
    CNN optimizada con filtros progresivos y BatchNormalization.

    Arquitectura:
        - GaussianNoise(0.01): Defensa adversaria (FocalX)
        - Bloque 1: Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
        - Bloque 2: Conv2D(64, 3x3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
        - Bloque 3: Conv2D(128, 3x3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
        - Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(num_classes, Softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Capa de suavizado de entrada (seguridad FocalX)
        layers.GaussianNoise(0.01),

        # --- Bloque Convolucional 1 (32 filtros) ---
        layers.Conv2D(32, (3, 3), padding='same', strides=1),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Bloque Convolucional 2 (64 filtros) ---
        layers.Conv2D(64, (3, 3), padding='same', strides=1),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Bloque Convolucional 3 (128 filtros) ---
        layers.Conv2D(128, (3, 3), padding='same', strides=1),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # --- Capas Densas ---
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    return model


def main():
    data_dir = os.environ.get(
        'DATA_DIR',
        os.path.join(os.getcwd(), 'detector_residuos', 'data', 'dataset-resized')
    )
    os.makedirs('models', exist_ok=True)

    # Carga con Data Augmentation activado
    train_ds, val_ds, class_names = prepare_datasets(data_dir, augment=True)
    print(f'Clases detectadas ({len(class_names)}): {class_names}')

    # Construcción del modelo optimizado
    model = build_model(num_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        # EarlyStopping con patience 15 (más margen con augmentation)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # ModelCheckpoint
        tf.keras.callbacks.ModelCheckpoint(
            'models/model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # ReduceLROnPlateau: reducir LR cuando val_loss se estanca
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    epochs = int(os.environ.get('EPOCHS', '70'))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Guardar metadatos
    with open('models/class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open('models/training_history.json', 'w', encoding='utf-8') as f:
        json.dump(hist_dict, f, ensure_ascii=False, indent=2)

    print(f'\nEntrenamiento completado. Modelo guardado en models/model.h5 (épocas={epochs})')
    print(f'Mejor val_loss: {min(history.history["val_loss"]):.4f}')
    print(f'Mejor val_accuracy: {max(history.history["val_accuracy"]):.4f}')


if __name__ == '__main__':
    main()
