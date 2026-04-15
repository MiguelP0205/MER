import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer

import gdown
import zipfile

MODEL_DIR = "model"
MODEL_ZIP_PATH = "model.zip"

def descargar_y_extraer_modelo():
    if not os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        url = "https://drive.google.com/uc?id=1l6vrPW5zFq6Yf9NFiBPvFgm0VSlJJnRU"

        print("Descargando modelo...")
        gdown.download(url, MODEL_ZIP_PATH, quiet=False)

        print("Extrayendo modelo...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")

        os.remove(MODEL_ZIP_PATH)
        print("Modelo listo ✅")

# ==========================
# CONFIGURACIÓN BÁSICA
# ==========================

# Carpeta donde guardaste el modelo con model.save("..."):
# IMPORTANTE: en Streamlit Cloud y GitHub, usa ruta relativa dentro del repo
MODEL_DIR = "model"

# Tokenizer usado en tu entrenamiento
TOKENIZER_NAME = "roberta-base"

# Constantes
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_FRAMES = 1000
AUDIO_DURATION_SECONDS = 30

MAX_LEN = 256

MOOD_LABELS = [
    "Aggressive/Hostile",
    "Calm/Peaceful",
    "Dark/Ominous",
    "Energetic/Upbeat",
    "Experimental/Eccentric",
    "Gritty/Raw",
    "Intense/Powerful",
    "Other",
    "Playful/Light",
    "Reflective",
    "Romantic/Intimate",
    "Sad/Melancholy",
    "Sentimental/Poignant",
    "Sophisticated/Refined",
    "Warm/Positive",
]

# ==========================
# CARGA (con cache para Streamlit)
# ==========================

# Streamlit cache_resource está en app.py, aquí dejamos loaders “puros”
def load_model():
    # Carga SavedModel (TF) aunque Keras 3 no lo soporte con load_model
    descargar_y_extraer_modelo()
    return tf.saved_model.load(MODEL_DIR)

def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ==========================
# PREPROCESADO
# ==========================

def load_audio_melspec(path: str, sr: int = SR) -> np.ndarray:
    y, _sr = librosa.load(path, sr=sr, mono=True)

    # Forzar duración fija
    target_len = sr * AUDIO_DURATION_SECONDS
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)
    S_db = S_db.T  # (T, n_mels)

    # Pad / recortar a MAX_FRAMES
    if S_db.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - S_db.shape[0]
        S_db = np.pad(S_db, ((0, pad_width), (0, 0)))
    else:
        S_db = S_db[:MAX_FRAMES, :]

    # Añadir canal
    S_db = S_db[..., np.newaxis]  # (T, n_mels, 1)

    # Normalizar por canción
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)

    return S_db.astype("float32")


def tokenize_lyrics_text(tokenizer, text: str):
    tokens = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    input_ids = tokens["input_ids"][0].astype("int32")
    attention_mask = tokens["attention_mask"][0].astype("int32")
    return input_ids, attention_mask


def load_lyrics_and_tokenize_from_file(tokenizer, path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return tokenize_lyrics_text(tokenizer, text)


# ==========================
# INFERENCIA + JSON
# ==========================

def predict_and_format_song(
    model,
    tokenizer,
    audio_file_path: str,
    lyric_file_path: str | None = None,
    lyric_text: str | None = None,
    title: str | None = None,
    artist: str | None = None,
    mood_threshold: float = 0.3,
) -> str:
    import os
    import json
    import tensorflow as tf

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_file_path}")

    if lyric_file_path is None and lyric_text is None:
        raise ValueError("Debes proporcionar lyric_file_path o lyric_text.")

    if lyric_file_path is not None and not os.path.exists(lyric_file_path):
        raise FileNotFoundError(f"Archivo de letra no encontrado: {lyric_file_path}")

    # ======================
    # 1) Preprocesar entradas
    # ======================
    melspec = load_audio_melspec(audio_file_path)

    if lyric_file_path is not None:
        input_ids, attention_mask = load_lyrics_and_tokenize_from_file(tokenizer, lyric_file_path)
        lyric_file_name = os.path.basename(lyric_file_path)
    else:
        input_ids, attention_mask = tokenize_lyrics_text(tokenizer, lyric_text)
        lyric_file_name = None

    # Batch
    melspec_batch = tf.expand_dims(melspec, axis=0)                 # (1, T, n_mels, 1)
    input_ids_batch = tf.expand_dims(input_ids, axis=0)             # (1, MAX_LEN)
    attention_mask_batch = tf.expand_dims(attention_mask, axis=0)   # (1, MAX_LEN)

    # Asegurar dtypes típicos
    melspec_batch = tf.convert_to_tensor(melspec_batch, dtype=tf.float32)
    input_ids_batch = tf.convert_to_tensor(input_ids_batch, dtype=tf.int32)
    attention_mask_batch = tf.convert_to_tensor(attention_mask_batch, dtype=tf.int32)

    # ======================
    # 2) Inferencia (SavedModel)
    # ======================
    if not hasattr(model, "signatures") or "serving_default" not in model.signatures:
        raise ValueError(
            "El modelo cargado no tiene la firma 'serving_default'. "
            "Verifica que lo cargas con tf.saved_model.load() y que exportaste un endpoint de serving."
        )

    infer = model.signatures["serving_default"]

    # Intento 1: usar nombres esperados (los que usabas en model.predict)
    try:
        outputs = infer(
            audio_melspec=melspec_batch,
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
        )
    except TypeError as e:
        # Si falla, es porque los nombres de entrada no coinciden.
        # Te damos un mensaje bien claro con los nombres reales.
        try:
            input_sig = infer.structured_input_signature
        except Exception:
            input_sig = "No disponible"

        raise TypeError(
            f"No pude llamar a 'serving_default' con los inputs "
            f"audio_melspec/input_ids/attention_mask.\n\n"
            f"Firma de entrada detectada:\n{input_sig}\n\n"
            f"Error original: {e}"
        )

    # outputs es un dict: {nombre_salida: tensor}
    out_keys = list(outputs.keys())
    if len(out_keys) < 2:
        raise ValueError(f"Se esperaban 2 salidas (moods y valence/arousal). Salidas recibidas: {out_keys}")

    # Orden estable por nombre para no depender del orden interno
    out_keys_sorted = sorted(out_keys)

    # Heurística: gusta identificar VA por su shape (1,2)
    moods_fused_pred = None
    va_fused_pred = None

    for k in out_keys:
        arr = outputs[k].numpy()
        if arr.ndim == 2 and arr.shape[-1] == 2:
            va_fused_pred = arr[0]
        else:
            # probable moods: (1, num_moods)
            if arr.ndim == 2 and arr.shape[0] == 1:
                moods_fused_pred = arr[0]

    # Fallback: si la heurística no pudo separar
    if moods_fused_pred is None or va_fused_pred is None:
        # intenta por orden
        a = outputs[out_keys_sorted[0]].numpy()
        b = outputs[out_keys_sorted[1]].numpy()
        moods_fused_pred = a[0]
        va_fused_pred = b[0]

        # Si aún no cuadra, avisar con shapes
        if not (isinstance(va_fused_pred, (list, tuple,)) or getattr(va_fused_pred, "shape", None) is not None):
            pass

    # Validación mínima
    if len(va_fused_pred) != 2:
        shapes = {k: tuple(outputs[k].shape) for k in out_keys}
        raise ValueError(
            f"No pude identificar correctamente la salida Valence/Arousal. "
            f"Shapes de salidas: {shapes}"
        )

    # ======================
    # 3) Emociones (Valence/Arousal)
    # ======================
    valence_norm = float(va_fused_pred[0])
    arousal_norm = float(va_fused_pred[1])

    if valence_norm >= 0 and arousal_norm >= 0:
        predicted_quadrant = "Q1"
        emotion_description = "Happy / Excited (High Valence, High Arousal)"
    elif valence_norm < 0 and arousal_norm >= 0:
        predicted_quadrant = "Q2"
        emotion_description = "Angry / Stressed (Low Valence, High Arousal)"
    elif valence_norm < 0 and arousal_norm < 0:
        predicted_quadrant = "Q3"
        emotion_description = "Sad / Depressed (Low Valence, Low Arousal)"
    else:
        predicted_quadrant = "Q4"
        emotion_description = "Calm / Relaxed (High Valence, Low Arousal)"

    emotions_output = {
        "valence_normalized": float(f"{valence_norm:.4f}"),
        "arousal_normalized": float(f"{arousal_norm:.4f}"),
        "predicted_quadrant": predicted_quadrant,
        "description": emotion_description,
    }

    # ======================
    # 4) Moods (filtrar + normalizar)
    # ======================
    moods_output = []
    selected_probs = []
    selected_labels = []

    for i, prob in enumerate(moods_fused_pred):
        if float(prob) >= mood_threshold:
            selected_probs.append(float(prob))
            # proteger si hay mismatch de longitudes
            label = MOOD_LABELS[i] if i < len(MOOD_LABELS) else f"mood_{i}"
            selected_labels.append(label)

    total = sum(selected_probs)
    if total > 0:
        for label, prob in zip(selected_labels, selected_probs):
            moods_output.append({
                "mood": label,
                "percentage": float(f"{(prob / total) * 100:.2f}")
            })

    moods_output = sorted(moods_output, key=lambda x: x["percentage"], reverse=True)

    # ======================
    # 5) JSON final
    # ======================
    audio_file_name = os.path.basename(audio_file_path)
    result_json = {
        "song_info": {
            "audio_file": audio_file_name,
            "lyric_file": lyric_file_name,
            "title": title if title else "Not provided",
            "artist": artist if artist else "Not provided",
        },
        "emotions": emotions_output,
        "moods_classification": moods_output,
    }

    return json.dumps(result_json, indent=2, ensure_ascii=False)
