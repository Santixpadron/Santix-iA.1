"""
Fase 4: Evaluación, Validación y Pruebas

Métricas generadas:
- Accuracy global (objetivo: >97%)
- Precision, Recall, F1-Score por clase
- Matriz de Confusión (visualización + guardado como imagen)

Enfoque: auditar falsas clasificaciones entre materiales transparentes
(vidrio vs. plástico) donde el ruido visual es mayor.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidores/CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from preprocess import prepare_datasets


def evaluate():
    """
    Evalúa el modelo entrenado sobre el conjunto de validación.
    Genera classification_report completo y matriz de confusión.
    """
    data_dir = os.environ.get(
        'DATA_DIR',
        os.path.join(os.getcwd(), 'data', 'dataset-resized')
    )
    models_dir = os.environ.get('MODELS_DIR', 'models')
    model_path = os.path.join(models_dir, 'model.h5')

    if not os.path.exists(model_path):
        print(f'ERROR: No se encontró el modelo en {model_path}')
        print('Ejecute primero: python train.py')
        return

    print(f'Cargando dataset desde: {data_dir}')
    _, val_ds, class_names = prepare_datasets(data_dir)

    print(f'Cargando modelo desde: {model_path}')
    model = tf.keras.models.load_model(model_path)

    # Recolectar predicciones sobre todo el conjunto de validación
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Métricas ---
    acc = accuracy_score(y_true, y_pred)
    print(f'\n{"="*60}')
    print(f'  RESULTADOS DE EVALUACIÓN - TrashNet CNN')
    print(f'{"="*60}')
    print(f'\n  Accuracy Global: {acc:.4f} ({acc*100:.2f}%)')
    print(f'  Objetivo: >97.88% ± 0.13%')
    print(f'\n{"="*60}')

    # Classification Report: Precision, Recall, F1-Score por clase
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Visualización de la Matriz de Confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Matriz de Confusión - TrashNet CNN', fontsize=14, fontweight='bold')
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Predicción del Modelo', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Guardar imagen
    output_path = os.path.join(models_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nMatriz de confusión guardada en: {output_path}')
    plt.close()

    # Análisis específico: falsas clasificaciones entre materiales transparentes
    if 'glass' in class_names and 'plastic' in class_names:
        glass_idx = class_names.index('glass')
        plastic_idx = class_names.index('plastic')
        glass_as_plastic = cm[glass_idx][plastic_idx]
        plastic_as_glass = cm[plastic_idx][glass_idx]
        print(f'\n  Análisis de materiales transparentes:')
        print(f'  - Vidrio clasificado como plástico: {glass_as_plastic}')
        print(f'  - Plástico clasificado como vidrio: {plastic_as_glass}')


if __name__ == '__main__':
    evaluate()
