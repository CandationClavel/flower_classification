"""API web para clasificar fotografías con el modelo entrenado."""

from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from class_names import CLASS_NAMES


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resultados" / "mejor_modelo.keras"
IMAGE_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontró el modelo en {MODEL_PATH}. "
        "Ejecuta primero el entrenamiento."
    )

model = tf.keras.models.load_model(MODEL_PATH)
app = FastAPI(title="Clasificador de flores", version="1.0.0")


def preprocess_image(contents: bytes) -> np.ndarray:
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="La imagen supera los 10 MB.")
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400, detail="El archivo no es una imagen válida."
        ) from error
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


@app.get("/", response_class=FileResponse)
def index() -> str:
    return str(BASE_DIR / "web" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | int | str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen.")
    contents = await file.read()
    probabilities = model.predict(preprocess_image(contents), verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": float(probabilities[class_index]),
    }
