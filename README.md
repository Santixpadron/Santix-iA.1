<div align="center">

# 🌿 Santix-iA.1: Detector de Residuos Inteligente

### Inteligencia Artificial para un Futuro Sostenible

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Activo-success?style=for-the-badge)](https://github.com/)

*Una solución de Deep Learning diseñada para automatizar la clasificación de residuos y fomentar el reciclaje eficiente.*

[Explorar Documentación](#-estructura-del-proyecto) • [Ver Demo](#-interfaz-de-usuario) • [Reportar Bug](https://github.com/)

</div>

---

## 📖 Descripción del Proyecto

**Santix-iA.1** es un sistema avanzado de clasificación de imágenes basado en Redes Neuronales Convolucionales (CNN). Entrenado con el dataset **TrashNet**, este modelo es capaz de identificar y categorizar residuos en 6 clases distintas: cartón, vidrio, metal, papel, plástico y basura general.

El objetivo principal es proporcionar una herramienta tecnológica que facilite la separación de residuos en orígen, integrable en aplicaciones móviles o sistemas de visión artificial para plantas de reciclaje.

### ✨ Características Principales

- **Arquitectura CNN Personalizada**: Optimizada para un equilibrio entre precisión y velocidad de inferencia.
- **Interfaz Web Intuitiva**: Desarrollada con Streamlit para pruebas rápidas y demostraciones en tiempo real.
- **Preprocesamiento Robusto**: Pipelines de datos que incluyen aumentación de imágenes para mejorar la generalización del modelo.
- **Código Modular**: Estructura profesional y organizada para facilitar la escalabilidad y el mantenimiento.

---

## 📂 Estructura del Proyecto

El repositorio está organizado siguiendo las mejores prácticas de ingeniería de software para proyectos de Machine Learning. A continuación, se detalla el contenido de cada directorio:

```bash
Santix-iA.1/
├── .streamlit/             # Configuración de la interfaz gráfica (tema, servidor, etc.)
│   └── config.toml
├── detector_residuos/      # NÚCLEO DEL PROYECTO
│   ├── data/               # Almacenamiento de datasets (imágenes crudas y procesadas)
│   ├── app.py              # Aplicación web principal (Streamlit) para inferencia
│   ├── evaluate.py         # Script para evaluar el rendimiento del modelo (métricas, matriz de confusión)
│   ├── preprocess.py       # Pipelines de carga, redimensionamiento y normalización de imágenes
│   ├── train.py            # Script de entrenamiento (definición de arquitectura CNN + loops de entrenamiento)
│   ├── utils.py            # Funciones auxiliares reutilizables (carga de modelos, manejo de archivos)
│   └── requirements.txt    # Lista de dependencias y librerías necesarias
├── models/                 # ARTEFACTOS GENERADOS
│   ├── model.h5            # Modelo entrenado final (pesos y arquitectura guardados)
│   ├── training_history.json # Registro de métricas durante el entrenamiento (loss, accuracy)
│   └── confusion_matrix.png  # Visualización del rendimiento por clase
├── .gitignore              # Archivos excluidos del control de versiones
└── README.md               # Documentación principal (este archivo)
```

---

## 🚀 Guía de Inicio Rápido

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Requisitos Previos

Asegúrate de tener instalado **Python 3.8** o superior.

### 2. Instalación

Clona el repositorio e instala las dependencias:

```bash
# Navegar al directorio del proyecto
cd Santix-iA.1/detector_residuos

# Instalar librerías
pip install -r requirements.txt
```

### 3. Uso

#### 🖥️ Ejecutar la Interfaz Web (Recomendado)
Para probar el modelo con tus propias imágenes mediante una interfaz amigable:

```bash
streamlit run app.py
```
*Esto abrirá una pestaña en tu navegador donde podrás subir imágenes y ver la predicción en tiempo real.*

#### 🧠 Entrenar el Modelo
Si deseas re-entrenar la red neuronal desde cero con nuevos datos:

```bash
python train.py
```
*Nota: Puedes ajustar los hiperparámetros (épocas, batch size) directamente en el script o mediante variables de entorno.*

#### 📊 Evaluar Rendimiento
Para generar reportes de clasificación y matrices de confusión sobre el set de validación:

```bash
python evaluate.py
```

---

## 🧠 Arquitectura Técnica

El modelo utiliza una **Red Neuronal Convolucional (CNN)** secuencial diseñada con TensorFlow/Keras:

1.  **Entrada**: Imágenes redimensionadas a 64x64 píxeles (RGB).
2.  **Extracción de Características**:
    -   3 Bloques convolucionales (Conv2D + ReLU + MaxPooling2D).
    -   Filtros progresivos (32 $\rightarrow$ 64 $\rightarrow$ 128) para capturar desde bordes simples hasta texturas complejas.
3.  **Regularización**: Capas `Dropout` para prevenir el sobreajuste (Overfitting).
4.  **Clasificación**:
    -   Capa densa (Fully Connected) de 512 neuronas.
    -   Capa de salida con activación `Softmax` para probabilidad de 6 clases.

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar la precisión del modelo o la interfaz:

1.  Haz un Fork del proyecto.
2.  Crea una rama para tu característica (`git checkout -b feature/NuevaMejora`).
3.  Haz Commit de tus cambios (`git commit -m 'Agregada nueva funcionalidad'`).
4.  Haz Push a la rama (`git push origin feature/NuevaMejora`).
5.  Abre un Pull Request.

---

<div align="center">

Creado con 💚 por [Tu Nombre/Usuario] para el cuidado del medio ambiente.

</div>
