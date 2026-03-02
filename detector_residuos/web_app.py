"""
Web App — Clasificador de Residuos AI
Backend Flask con API REST para inferencia del modelo CNN.
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify

# Agregar directorio padre para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_preprocess_image_bytes, load_model, load_class_names

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Cargar modelo al inicio
print("🔄 Cargando modelo CNN...")
try:
    model = load_model()
    class_names = load_class_names()
    print(f"✅ Modelo cargado. Clases: {class_names}")
except Exception as e:
    print(f"⚠️  Error cargando modelo: {e}")
    model = None
    class_names = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado. Ejecute python train.py primero.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío.'}), 400

    image_bytes = file.read()
    x = load_and_preprocess_image_bytes(image_bytes)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))

    # Mapeo de clases a iconos y colores
    class_meta = {
        'cardboard': {'icon': '📦', 'color': '#D4915E', 'label': 'Cartón'},
        'glass':     {'icon': '🍶', 'color': '#7EC8E3', 'label': 'Vidrio'},
        'metal':     {'icon': '🥫', 'color': '#A8A8A8', 'label': 'Metal'},
        'paper':     {'icon': '📄', 'color': '#F5E6CA', 'label': 'Papel'},
        'plastic':   {'icon': '🧴', 'color': '#FF6B6B', 'label': 'Plástico'},
        'trash':     {'icon': '🗑️', 'color': '#6C757D', 'label': 'Residuo General'},
    }

    results = []
    for i, name in enumerate(class_names):
        meta = class_meta.get(name, {'icon': '❓', 'color': '#888', 'label': name})
        results.append({
            'class': name,
            'label': meta['label'],
            'icon': meta['icon'],
            'color': meta['color'],
            'probability': round(float(preds[0][i]) * 100, 2),
        })

    results.sort(key=lambda x: x['probability'], reverse=True)

    prediction = class_meta.get(class_names[idx], {'icon': '❓', 'color': '#888', 'label': class_names[idx]})

    return jsonify({
        'prediction': {
            'class': class_names[idx],
            'label': prediction['label'],
            'icon': prediction['icon'],
            'color': prediction['color'],
            'confidence': round(float(preds[0][idx]) * 100, 2),
        },
        'probabilities': results,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
