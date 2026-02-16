# Detector de Residuos — CNN Personalizada (TrashNet)

Sistema de clasificación automática de residuos mediante una Red Neuronal Convolucional (CNN) ligera, entrenada con el dataset **TrashNet** (6 clases: cardboard, glass, metal, paper, plastic, trash).

## Arquitectura del Modelo

| Capa | Configuración |
|------|---------------|
| GaussianNoise | σ = 0.01 (defensa adversaria FocalX) |
| Conv2D × 3 bloques | 32 filtros, kernel 2×2, padding same, ReLU |
| MaxPooling2D × 3 | Pool size 2×2 |
| Dropout × 3 | Rate 0.2 |
| Flatten | Transición a vector denso |
| Dense | 512 neuronas, ReLU |
| Dense (salida) | 6 neuronas, Softmax |

**Input:** 64×64×3 · **Optimizador:** Adam · **Pérdida:** Categorical Cross-Entropy

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Resolución | 64×64 px |
| Split train/val | 75% / 25% |
| Épocas | 70 (con EarlyStopping) |
| Batch size | 32 |
| EarlyStopping patience | 10 (monitor: val_loss) |
| ModelCheckpoint | Mejor modelo por val_loss |

**Objetivo de Accuracy:** >97.88% ± 0.13%

## Estructura del Proyecto

```
detector_residuos/
├── data/dataset-resized/   # Dataset TrashNet (6 subcarpetas por clase)
├── models/                 # Modelo entrenado (.h5) + metadatos
├── preprocess.py           # Fase 1: Carga y preprocesamiento de datos
├── train.py                # Fase 2+3: Arquitectura CNN + Pipeline de entrenamiento
├── evaluate.py             # Fase 4: Evaluación y métricas
├── app.py                  # Interfaz Streamlit para inferencia
├── utils.py                # Utilidades de carga e inferencia
└── requirements.txt        # Dependencias
```

## Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Entrenamiento
```bash
python train.py
```
Variables de entorno opcionales:
- `DATA_DIR`: Ruta al dataset (default: `detector_residuos/data/dataset-resized`)
- `EPOCHS`: Número de épocas (default: `70`)

### Evaluación
```bash
python evaluate.py
```
Genera: classification report (Precision, Recall, F1-Score) + `models/confusion_matrix.png`

### Interfaz de inferencia (Streamlit)
```bash
streamlit run app.py
```

## Commits

| Fase | Mensaje |
|------|---------|
| 1 | `feat: implement data loading and preprocessing for TrashNet` |
| 2 | `feat: define CNN architecture with Conv2D and Dropout layers` |
| 3 | `fix: optimize training pipeline with Adam and EarlyStopping` |
| 4 | `docs: update evaluation metrics and confusion matrix results` |

## Referencias

- Goodfellow, I. J. (2016). *Deep Learning*. MIT Press.
- Barcelona Fernández, O. (2024). *Deep learning. Redes neuronales convolucionales*. Univ. Zaragoza.
- FocalX - AI (2025). *Ataques Adversarios a la IA*.
