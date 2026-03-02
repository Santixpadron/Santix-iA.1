# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec para Eco-Detector AI.
Genera un ejecutable que incluye el modelo CNN, templates y utilidades.

Uso:
    pyinstaller eco_detector.spec
"""

import os

block_cipher = None

# Rutas - SPECPATH es la carpeta donde está el .spec (detector_residuos/)
spec_dir = SPECPATH
models_dir = os.path.join(spec_dir, '..', 'models')

a = Analysis(
    ['launcher.py'],
    pathex=[spec_dir],
    binaries=[],
    datas=[
        # Templates HTML
        (os.path.join(spec_dir, 'templates', 'index.html'), 'templates'),
        # Modelo y clases
        (os.path.join(models_dir, 'model.h5'), 'models'),
        (os.path.join(models_dir, 'class_names.json'), 'models'),
        # Utilidades de inferencia
        (os.path.join(spec_dir, 'utils.py'), '.'),
    ],
    hiddenimports=[
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.eager',
        'tensorflow.lite.python.lite',
        'numpy',
        'PIL',
        'PIL.Image',
        'flask',
        'flask.json',
        'jinja2',
        'markupsafe',
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.debug',
        'click',
        'itsdangerous',
        'h5py',
        'google.protobuf',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'seaborn',
        'pandas',
        'streamlit',
        'cv2',
        'sklearn',
        'scipy.spatial.cKDTree',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EcoDetectorAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='EcoDetectorAI',
)
