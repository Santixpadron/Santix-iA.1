"""
Aplicación Streamlit para inferencia del clasificador de residuos.
Utiliza el modelo CNN entrenado para predecir la clase de residuo
a partir de una imagen cargada por el usuario.
"""

import streamlit as st
from utils import load_and_preprocess_image_bytes, load_model, load_class_names
import numpy as np


st.set_page_config(
    page_title='Clasificador de Residuos AI',
    page_icon='♻️',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #00cc66;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00aa55;
    }
    .css-1v0mbdj.etr89bj1 {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1 {
        color: #00cc66;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stProgress .st-bo {
        background-color: #00cc66;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title('♻️ Eco-Detector AI')
    st.markdown('<p style="text-align: center; color: #888;">Sube una foto y la IA clasificará el residuo en segundos.</p>', unsafe_allow_html=True)

uploaded = st.file_uploader('', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

if uploaded is not None:
    # Centrar imagen
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        bytes_data = uploaded.read()
        st.image(bytes_data, use_container_width=True, caption='📸 Imagen analizada')

    # Inferencia
    try:
        model = load_model()
        classes = load_class_names()
    except FileNotFoundError:
        st.error('⚠️ Modelo no encontrado. Entrena primero.')
        model = None
        classes = None

    if model is not None:
        with st.spinner('🤖 Analizando átomos...'):
            x = load_and_preprocess_image_bytes(bytes_data)
            preds = model.predict(x)
            idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][idx])
            label = classes[idx].upper()

        # Resultados
        st.divider()
        st.markdown(f"<h2 style='text-align: center;'>Predicción: <span style='color: #00cc66;'>{label}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confianza: <b>{confidence*100:.1f}%</b></p>", unsafe_allow_html=True)

        st.caption('Probabilidades detalladas:')
        for c, p in sorted(zip(classes, preds[0]), key=lambda x: x[1], reverse=True):
            col_lbl, col_bar, col_val = st.columns([2, 6, 2])
            with col_lbl: st.write(f"**{c.title()}**")
            with col_bar: st.progress(float(p))
            with col_val: st.write(f"{p*100:.1f}%")
