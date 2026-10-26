# Clasificacion de flores con transferencia de aprendizaje

Proyecto para clasificar las 104 clases del dataset **100 Flowers** usado en
`https://www.kaggle.com/competitions/flower-classification-with-tpus/data`. Los TFRecord se leen desde
`tfrecords-jpeg-224x224/` y conservan la separacion `train`, `val` y `test` de
la referencia.

## Enfoque

El modelo usa `EfficientNetB0` con pesos preentrenados en ImageNet:

1. Se congela la red convolucional y se entrena una nueva cabeza de 104 clases.
2. Se descongelan las ultimas 30 capas (manteniendo Batch Normalization
   congelada) y se hace fine-tuning con una tasa de aprendizaje menor.
3. Se generan las curvas de accuracy/loss y la matriz de confusion sobre
   validacion. El split `test` de este dataset no tiene etiquetas, por lo que
   no puede usarse para una matriz de confusion.

## Ejecucion local o en YUCA

```bash
conda env create -f flores-tl.yml
conda activate flores-tl
python clasificacion_flores_tl.py \
  --data-dir . \
  --output-dir resultados \
  --epochs 15 \
  --fine-tune-epochs 5 \
  --batch-size 32
```

En YUCA, el archivo [script.slrm](script.slrm) asume que el entorno ya fue
creado y que, para mi caso en particular, `DATASET_DIR` apunta a este directorio:

```bash
conda env create -f flores-tl.yml
export DATASET_DIR=/lustre/home/estudiante_11/TL
sbatch script.slrm
```

El job guarda logs en `logs/`. El entrenamiento genera en `resultados/`:

- `modelo_flores_tl.keras`: modelo final.
- `mejor_modelo.keras`: mejor checkpoint por `val_accuracy`.
- `accuracy_loss.png`: curvas de entrenamiento y validacion.
- `matriz_confusion.png`: matriz de confusion de las 104 clases.
- `matriz_confusion.txt`: precision, recall y F1 por clase.

El repositorio incluye únicamente los TFRecord de `tfrecords-jpeg-224x224/`
para mantener un tamaño razonable. Las carpetas `tfrecords-jpeg-192x192/`,
`tfrecords-jpeg-331x331/` y `tfrecords-jpeg-512x512/` están excluidas mediante
`.gitignore`; pueden conservarse localmente o descargarse por separado si se
necesita entrenar con otra resolución.

## Aplicacion web

La API y la interfaz web reutilizan `resultados/mejor_modelo.keras`.
Instala las dependencias (ya incluidas en `flores-tl.yml`) y arranca el
servidor desde la raiz del proyecto:

```bash
pip install -r requirements-web.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Abre `http://localhost:8000`. En un celular, abre la URL del equipo en la
misma red: `http://IP.DEL.EQUIPO:8000`. La interfaz ofrece botones separados para seleccionar una imagen
existente o tomar una fotografia con la camara trasera. La API tambien expone
`GET /health` y `POST /predict`.

El archivo [exportar_modelo.py](exportar_modelo.py) exporta opcionalmente el
modelo a `modelo_exportado/` en formato SavedModel:

```bash
python exportar_modelo.py
```