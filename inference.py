import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFRobertaModel, TFRobertaMainLayer
import gdown

# ==========================
# CONFIGURACIÓN BÁSICA Y DESCARGA
# ==========================

# Nombre del archivo local donde se guardará el modelo
MODEL_PATH = "my_model.keras"

# ID del archivo my_model.keras en Google Drive
DRIVE_FILE_ID = "1onGKhJcSQamYZsdJZuC9OKixT5igAaRZ"

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        print("Descargando modelo (.keras)...")
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Modelo listo ✅")

# Tokenizer usado en tu entrenamiento
TOKENIZER_NAME = "roberta-base"

# Constantes de audio y texto
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_FRAMES = 1000
AUDIO_DURATION_SECONDS = 30

MAX_LEN = 256

MOOD_LABELS = [
    "Happy", 
    "Elated", 
    "Enthusiastic", 
    "Excited/Alert",
    "Tense", 
    "Nervous", 
    "Stressed", 
    "Upset",
    "Sad", 
    "Depressed", 
    "Sluggish", 
    "Bored",
    "Calm", 
    "Relaxed", 
    "Serene", 
    "Contented"
]

# ==========================
# CARGA DEL MODELO Y TOKENIZER
# ==========================

def load_model():
    descargar_modelo()
    
    # Diccionario con las capas personalizadas de HuggingFace RoBERTa
    custom_objects = {
        "TFRobertaModel": TFRobertaModel,
        "TFRobertaMainLayer": TFRobertaMainLayer,
    }
    
    return tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objects, 
        compile=False
    )

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
# INFERENCIA + FORMATO JSON
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

    # Lote de dimensión (1, ...)
    melspec_batch = tf.expand_dims(melspec, axis=0)                 # (1, T, n_mels, 1)
    input_ids_batch = tf.expand_dims(input_ids, axis=0)             # (1, MAX_LEN)
    attention_mask_batch = tf.expand_dims(attention_mask, axis=0)   # (1, MAX_LEN)

    # Convertir a tensores
    melspec_batch = tf.convert_to_tensor(melspec_batch, dtype=tf.float32)
    input_ids_batch = tf.convert_to_tensor(input_ids_batch, dtype=tf.int32)
    attention_mask_batch = tf.convert_to_tensor(attention_mask_batch, dtype=tf.int32)

    # ======================
    # 2) Inferencia con modelo Keras
    # ======================
    # Intentamos pasar por nombre de entrada, si falla, pasamos por lista ordenada
    inputs_dict = {
        "audio_melspec": melspec_batch,
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
    }

    try:
        raw_outputs = model(inputs_dict, training=False)
    except Exception:
        inputs_list = [melspec_batch, input_ids_batch, attention_mask_batch]
        raw_outputs = model(inputs_list, training=False)

    # Normalizar las salidas recibidas a una lista de arrays NumPy
    if isinstance(raw_outputs, dict):
        output_list = [v.numpy() for v in raw_outputs.values()]
    elif isinstance(raw_outputs, (list, tuple)):
        output_list = [v.numpy() for v in raw_outputs]
    else:
        output_list = [raw_outputs.numpy()]

    moods_fused_pred = None
    va_fused_pred = None

    # Identificar salidas por forma (shape)
    for arr in output_list:
        if arr.ndim == 2 and arr.shape[-1] == 2:
            va_fused_pred = arr[0]
        elif arr.ndim == 2 and arr.shape[-1] == len(MOOD_LABELS):
            moods_fused_pred = arr[0]

    # Fallback posicional si no se identificaron por forma
    if moods_fused_pred is None or va_fused_pred is None:
        if len(output_list) >= 2:
            # Asumimos que la primera salida o la segunda corresponden
            moods_fused_pred = output_list[0][0]
            va_fused_pred = output_list[1][0]
        else:
            raise ValueError(f"No se pudieron extraer 2 salidas del modelo Keras. Recibidas: {len(output_list)}")

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