import os
import sys
import zipfile

# Configuración de entorno
os.environ["USE_TF"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import tempfile
import numpy as np
import librosa

import tensorflow as tf
from transformers import AutoTokenizer, TFRobertaModel
import gdown

# =========================================================
# CONFIGURACIÓN BÁSICA Y DESCARGA
# =========================================================

MODEL_PATH = "my_model.keras"
DRIVE_FILE_ID = "1onGKhJcSQamYZsdJZuC9OKixT5igAaRZ"

TOKENIZER_NAME = "roberta-base"

SR = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_FRAMES = 1000
AUDIO_DURATION_SECONDS = 30
MAX_LEN = 256

MOOD_LABELS = [
    "Happy", "Elated", "Enthusiastic", "Excited/Alert",
    "Tense", "Nervous", "Stressed", "Upset",
    "Sad", "Depressed", "Sluggish", "Bored",
    "Calm", "Relaxed", "Serene", "Contented"
]

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        print("Descargando modelo (.keras)...")
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Modelo listo ✅")

# =========================================================
# CONSTRUCCIÓN EXPLICITA DE ARQUITECTURA (COLAB)
# =========================================================

def build_multimodal_model():
    """Reconstruye la arquitectura exacta declarada en Colab."""
    num_moods = len(MOOD_LABELS)

    # Entradas
    audio_input = tf.keras.Input(shape=(MAX_FRAMES, N_MELS, 1), name="audio_melspec")
    input_ids_in = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    att_mask_in  = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

    # --- Rama Audio ---
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(audio_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    new_frames = MAX_FRAMES // 4  # 250
    new_mels = N_MELS // 4        # 32

    x = tf.keras.layers.Reshape((new_frames, new_mels * 64))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    audio_emb = tf.keras.layers.Dense(256, activation="relu", name="audio_emb")(x)

    va_audio   = tf.keras.layers.Dense(2, activation="tanh", name="va_audio")(audio_emb)
    moods_audio = tf.keras.layers.Dense(num_moods, activation="sigmoid", name="moods_audio")(audio_emb)

    # --- Rama Texto ---
    text_encoder = TFRobertaModel.from_pretrained(TOKENIZER_NAME)
    text_outputs = text_encoder(input_ids=input_ids_in, attention_mask=att_mask_in)
    cls_token = text_outputs.last_hidden_state[:, 0, :]

    t = tf.keras.layers.Dense(256, activation="relu")(cls_token)
    t = tf.keras.layers.Dropout(0.5)(t)

    text_emb = tf.keras.layers.Dense(256, activation="relu", name="text_emb")(t)

    va_text    = tf.keras.layers.Dense(2, activation="tanh", name="va_text")(text_emb)
    moods_text = tf.keras.layers.Dense(num_moods, activation="sigmoid", name="moods_text")(text_emb)

    # --- Fusión ---
    fusion_concat = tf.keras.layers.Concatenate(name="fusion_concat")([audio_emb, text_emb])
    f = tf.keras.layers.Dense(256, activation="relu")(fusion_concat)
    f = tf.keras.layers.Dropout(0.5)(f)
    fusion_emb = tf.keras.layers.Dense(128, activation="relu", name="fusion_emb")(f)

    va_fused    = tf.keras.layers.Dense(2, activation="tanh", name="va_fused")(fusion_emb)
    moods_fused = tf.keras.layers.Dense(num_moods, activation="sigmoid", name="moods_fused")(fusion_emb)

    # Modelo completo
    model = tf.keras.Model(
        inputs=[audio_input, input_ids_in, att_mask_in],
        outputs=[moods_fused, va_fused, moods_audio, moods_text, va_audio, va_text],
        name="multimodal_emotion_mood_model"
    )
    return model

# =========================================================
# CARGA DEL MODELO Y TOKENIZER
# =========================================================

def load_model():
    descargar_modelo()
    print("Construyendo la arquitectura del modelo...")
    model = build_multimodal_model()

    print("Cargando pesos entrenados...")
    try:
        model.load_weights(MODEL_PATH)
        print("Pesos cargados directamente desde .keras ✅")
    except Exception as e:
        print(f"Extrayendo archivo de pesos por compatibilidad: {e}")
        extracted_dir = "extracted_weights"
        os.makedirs(extracted_dir, exist_ok=True)
        with zipfile.ZipFile(MODEL_PATH, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

        weights_file = None
        for root, dirs, files in os.walk(extracted_dir):
            for f in files:
                if f.endswith(".h5") or f.endswith(".weights.h5"):
                    weights_file = os.path.join(root, f)
                    break

        if weights_file:
            model.load_weights(weights_file)
            print("Pesos cargados con éxito desde archivo .h5 interno ✅")
        else:
            model.load_weights(MODEL_PATH, by_name=True)

    return model

def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# =========================================================
# PREPROCESADO DE AUDIO Y TEXTO
# =========================================================

def load_audio_melspec(path: str, sr: int = SR) -> np.ndarray:
    y, _sr = librosa.load(path, sr=sr, mono=True)

    target_len = sr * AUDIO_DURATION_SECONDS
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = S_db.T

    if S_db.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - S_db.shape[0]
        S_db = np.pad(S_db, ((0, pad_width), (0, 0)))
    else:
        S_db = S_db[:MAX_FRAMES, :]

    S_db = S_db[..., np.newaxis]
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

# =========================================================
# INFERENCIA Y FORMATO JSON
# =========================================================

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

    melspec = load_audio_melspec(audio_file_path)

    if lyric_file_path is not None:
        input_ids, attention_mask = load_lyrics_and_tokenize_from_file(tokenizer, lyric_file_path)
        lyric_file_name = os.path.basename(lyric_file_path)
    else:
        input_ids, attention_mask = tokenize_lyrics_text(tokenizer, lyric_text)
        lyric_file_name = None

    melspec_batch = tf.expand_dims(melspec, axis=0)
    input_ids_batch = tf.expand_dims(input_ids, axis=0)
    attention_mask_batch = tf.expand_dims(attention_mask, axis=0)

    inputs = [melspec_batch, input_ids_batch, attention_mask_batch]

    raw_outputs = model(inputs, training=False)

    if isinstance(raw_outputs, (list, tuple)):
        output_list = [v.numpy() if hasattr(v, "numpy") else np.array(v) for v in raw_outputs]
    elif isinstance(raw_outputs, dict):
        output_list = [v.numpy() if hasattr(v, "numpy") else np.array(v) for v in raw_outputs.values()]
    else:
        output_list = [raw_outputs.numpy()]

    moods_fused_pred = output_list[0][0]
    va_fused_pred = output_list[1][0]

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