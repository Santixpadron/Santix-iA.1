"""
Launcher — Eco-Detector AI
Inicia el servidor Flask y abre automáticamente el navegador.
Diseñado para funcionar tanto en desarrollo como empaquetado con PyInstaller.
"""

import os
import sys
import socket
import threading
import webbrowser
import time


def get_base_path():
    """Retorna la ruta base, compatible con PyInstaller."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def get_free_port():
    """Encuentra un puerto TCP libre."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def open_browser(port):
    """Abre el navegador tras un breve delay para dar tiempo al servidor."""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{port}')


def main():
    base_path = get_base_path()

    # Configurar rutas de recursos
    template_dir = os.path.join(base_path, 'templates')
    models_dir = os.path.join(base_path, 'models')

    # Verificar que existen los archivos necesarios
    model_path = os.path.join(models_dir, 'model.h5')
    if not os.path.exists(model_path):
        print(f'❌ No se encontró el modelo en: {model_path}')
        print('   Asegúrate de que model.h5 esté en la carpeta models/')
        input('Presiona Enter para salir...')
        sys.exit(1)

    if not os.path.exists(template_dir):
        print(f'❌ No se encontró la carpeta templates en: {template_dir}')
        input('Presiona Enter para salir...')
        sys.exit(1)

    # Importar después de validar para evitar delay innecesario si falta algo
    print('🔄 Cargando dependencias...')

    # Suprimir logs excesivos de TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    import numpy as np
    from flask import Flask, render_template, request, jsonify

    # Agregar base_path al sys.path para encontrar utils
    sys.path.insert(0, base_path)
    from utils import load_and_preprocess_image_bytes

    # Importar TensorFlow y cargar modelo
    import tensorflow as tf
    import json

    print('🔄 Cargando modelo CNN...')
    try:
        model = tf.keras.models.load_model(model_path)
        class_names_path = os.path.join(models_dir, 'class_names.json')
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f'✅ Modelo cargado. Clases: {class_names}')
    except Exception as e:
        print(f'❌ Error cargando modelo: {e}')
        input('Presiona Enter para salir...')
        sys.exit(1)

    # Crear app Flask
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=os.path.join(base_path, 'static'))

    # Mapeo de clases a iconos y colores
    class_meta = {
        'cardboard': {'icon': '📦', 'color': '#D4915E', 'label': 'Cartón'},
        'glass':     {'icon': '🍶', 'color': '#7EC8E3', 'label': 'Vidrio'},
        'metal':     {'icon': '🥫', 'color': '#A8A8A8', 'label': 'Metal'},
        'paper':     {'icon': '📄', 'color': '#F5E6CA', 'label': 'Papel'},
        'plastic':   {'icon': '🧴', 'color': '#FF6B6B', 'label': 'Plástico'},
        'trash':     {'icon': '🗑️', 'color': '#6C757D', 'label': 'Residuo General'},
    }

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Archivo vacío.'}), 400

        image_bytes = file.read()
        x = load_and_preprocess_image_bytes(image_bytes)
        preds = model.predict(x, verbose=0)
        idx = int(np.argmax(preds[0]))

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

        prediction = class_meta.get(class_names[idx],
                                     {'icon': '❓', 'color': '#888', 'label': class_names[idx]})

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

    # Encontrar puerto libre e iniciar
    port = get_free_port()

    print(f'\n♻️  Eco-Detector AI')
    print(f'   Servidor: http://localhost:{port}')
    print(f'   Abriendo navegador...')
    print(f'   Presiona Ctrl+C para detener el servidor.\n')

    # Abrir navegador en hilo separado
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Iniciar servidor Flask (sin reloader para evitar doble ejecución)
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
