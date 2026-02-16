"""
Aplicación Streamlit para inferencia del clasificador de residuos.
Utiliza el modelo CNN entrenado para predecir la clase de residuo
a partir de una imagen cargada por el usuario.
"""

import streamlit as st
from utils import load_and_preprocess_image_bytes, load_model, load_class_names
import numpy as np


st.set_page_config(
    page_title='Clasificador de Residuos - TrashNet CNN',
    page_icon='♻️',
    layout='centered'
)

st.title('♻️ Clasificador de Residuos')
st.markdown('**Modelo:** CNN personalizada entrenada con TrashNet (64x64)')
st.markdown('Carga una imagen y el modelo predirá la clase de residuo.')

st.divider()

uploaded = st.file_uploader('Selecciona una imagen', type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    bytes_data = uploaded.read()
    st.image(bytes_data, caption='Imagen cargada', use_container_width=True)

    try:
        model = load_model()
        classes = load_class_names()
    except FileNotFoundError:
        st.error('⚠️ No se encontró el modelo. Entrene primero ejecutando: `python train.py`')
        model = None
        classes = None

    if model is not None:
        x = load_and_preprocess_image_bytes(bytes_data)
        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])

        st.divider()

        if classes:
            st.success(f'**Predicción:** {classes[idx].upper()} ({confidence*100:.1f}%)')

            st.subheader('Probabilidades por clase:')
            for c, p in sorted(zip(classes, preds[0]), key=lambda x: x[1], reverse=True):
                st.progress(float(p), text=f'{c}: {p*100:.1f}%')
        else:
            st.write(f'Predicción (índice): {idx} ({confidence*100:.1f}%)')
