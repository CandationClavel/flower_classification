"""Exporta el modelo Keras al formato SavedModel para despliegues."""

from pathlib import Path

import tensorflow as tf


model_path = Path("resultados/modelo_flores_tl.keras")
export_path = Path("modelo_exportado")

if not model_path.is_file():
    raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

model = tf.keras.models.load_model(model_path)
model.export(export_path)
print(f"Modelo exportado en {export_path.resolve()}")
