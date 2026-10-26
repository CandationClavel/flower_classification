"""Clasificacion de las 104 clases de Flowers con transferencia de aprendizaje.

El dataset es el conjunto de TFRecord del notebook de referencia de Kaggle.
La matriz de confusion se calcula sobre validacion porque el split test no
contiene etiquetas.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from class_names import CLASS_NAMES


AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    """Define los parámetros configurables para entrenar y evaluar el modelo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def files_for_split(data_dir: Path, split: str, image_size: int) -> list[str]:
    """Obtiene los TFRecord del split y tamaño de imagen solicitados."""
    pattern = data_dir / f"tfrecords-jpeg-{image_size}x{image_size}" / split / "*.tfrec"
    files = sorted(str(path) for path in pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(f"No se encontraron TFRecord para {split}: {pattern}")
    return files


def count_records(filenames: list[str]) -> int:
    """Suma los ejemplos indicados en los nombres de los archivos TFRecord."""
    total = 0
    for filename in filenames:
        match = re.search(r"-([0-9]+)\.tfrec$", filename)
        if match is None:
            raise ValueError(f"No se pudo obtener el numero de ejemplos de {filename}")
        total += int(match.group(1))
    return total


def decode_labeled(example: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Convierte un registro serializado en una imagen y su clase numérica."""
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(parsed["image"], channels=3)
    image = tf.reshape(image, [image_size, image_size, 3])
    label = tf.cast(parsed["class"], tf.int32)
    return image, label


def load_dataset(
    filenames: list[str], image_size: int, batch_size: int, training: bool
) -> tf.data.Dataset:
    """Construye el pipeline eficiente de lectura, aumento y batching."""
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(
        lambda record: decode_labeled(record, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    if training:
        # El aumento se aplica solo al entrenamiento para mejorar la generalización.
        dataset = dataset.shuffle(2048, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda image, label: (tf.image.random_flip_left_right(image), label),
            num_parallel_calls=AUTOTUNE,
        )
        dataset = dataset.repeat()
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_model(image_size: int, trainable_tail: int = 0) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Crea EfficientNetB0 y una cabeza nueva para las 104 flores."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    # Primera etapa: base congelada; segunda etapa: ajuste fino de su parte final.
    base.trainable = trainable_tail > 0
    if trainable_tail > 0:
        for layer in base.layers[:-trainable_tail]:
            layer.trainable = False
        for layer in base.layers:
            # Batch Normalization se mantiene estable durante el fine-tuning.
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.RandomRotation(0.08)(inputs)
    x = tf.keras.layers.RandomZoom(0.15)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3 if trainable_tail == 0 else 1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def plot_history(histories: list[tf.keras.callbacks.History], output: Path) -> None:
    """Guarda las curvas de accuracy y loss de ambas etapas de entrenamiento."""
    history = {
        key: sum((h.history.get(key, []) for h in histories), [])
        for key in ("loss", "val_loss", "accuracy", "val_accuracy")
    }
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["accuracy"], label="entrenamiento")
    axes[0].plot(epochs, history["val_accuracy"], label="validacion")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[1].plot(epochs, history["loss"], label="entrenamiento")
    axes[1].plot(epochs, history["val_loss"], label="validacion")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def evaluate_and_plot(
    model: tf.keras.Model,
    validation_filenames: list[str],
    image_size: int,
    batch_size: int,
    validation_count: int,
    output: Path,
) -> None:
    """Evalúa validación y guarda la matriz de confusión y el reporte."""
    # Se vuelve a leer el split en el mismo orden para alinear etiquetas y predicciones.
    validation = load_dataset(
        validation_filenames,
        image_size,
        batch_size,
        training=False,
    )
    labels = np.concatenate(
        [batch_labels.numpy() for _, batch_labels in validation]
    )[:validation_count]
    validation = load_dataset(
        validation_filenames,
        image_size,
        batch_size,
        training=False,
    )
    predictions = model.predict(validation, steps=math.ceil(validation_count / batch_size), verbose=1)
    predicted_labels = np.argmax(predictions[: len(labels)], axis=1)
    matrix = confusion_matrix(labels, predicted_labels, labels=np.arange(NUM_CLASSES))
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Matriz de confusion - validacion",
        xlabel="Clase predicha",
        ylabel="Clase real",
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
    )
    axis.tick_params(axis="both", labelsize=5)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    report = classification_report(
        labels,
        predicted_labels,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    output.with_suffix(".txt").write_text(report, encoding="utf-8")


def main() -> None:
    """Ejecuta entrenamiento, fine-tuning, evaluación y guardado de resultados."""
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # El split test no tiene etiquetas; por eso train y val son los usados aquí.
    train_files = files_for_split(args.data_dir, "train", args.image_size)
    validation_filenames = files_for_split(args.data_dir, "val", args.image_size)
    train_count = count_records(train_files)
    validation_count = count_records(validation_filenames)
    image_size = args.image_size
    batch_size = args.batch_size
    train = load_dataset(train_files, image_size, batch_size, training=True)
    validation = load_dataset(validation_filenames, image_size, batch_size, training=False)
    steps = math.ceil(train_count / batch_size)
    validation_steps = math.ceil(validation_count / batch_size)

    # Etapa 1: entrena únicamente la cabeza sobre las características de ImageNet.
    model, _ = build_model(image_size)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "mejor_modelo.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=2),
    ]
    histories = [
        model.fit(
            train,
            validation_data=validation,
            steps_per_epoch=steps,
            validation_steps=validation_steps,
            epochs=args.epochs,
            callbacks=callbacks,
        )
    ]

    # Etapa 2: descongela las últimas capas y continúa con una tasa menor.
    model, _ = build_model(image_size, args.fine_tune_layers)
    model.load_weights(args.output_dir / "mejor_modelo.keras")
    histories.append(
        model.fit(
            train,
            validation_data=validation,
            steps_per_epoch=steps,
            validation_steps=validation_steps,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
    )
    # El modelo final se puede cargar directamente desde la API web.
    model.save(args.output_dir / "modelo_flores_tl.keras")
    plot_history(histories, args.output_dir / "accuracy_loss.png")
    evaluate_and_plot(
        model,
        validation_filenames,
        image_size,
        batch_size,
        validation_count,
        args.output_dir / "matriz_confusion.png",
    )
    print(f"Resultados guardados en {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
