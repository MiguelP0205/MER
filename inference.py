import os
import sys

# Forzar configuración de entorno
os.environ["USE_TF"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import tempfile
import importlib
import numpy as np
import librosa

import tensorflow as tf
import keras
from transformers import AutoTokenizer
import gdown

# Importar explícitamente el módulo TensorFlow de RoBERTa
import transformers
try:
    import transformers.models.roberta.modeling_tf_roberta as roberta_mod
except Exception:
    roberta_mod = None

# =========================================================
# REGISTRO GLOBAL Y DINÁMICO DE CUSTOM OBJECTS
# =========================================================
GLOBAL_CUSTOM_OBJECTS = {}

def build_custom_objects_registry():
    """Escanea y registra todas las clases disponibles en Keras y Transformers."""
    modules_to_scan = [
        transformers,
        roberta_mod,
        keras.layers,
        tf.keras.layers,
    ]

    try:
        import tf_keras
        modules_to_scan.append(tf_keras.layers)
    except Exception:
        pass

    for mod in modules_to_scan:
        if mod is None:
            continue
        for name, cls in getattr(mod, "__dict__", {}).items():
            if isinstance(cls, type):
                GLOBAL_CUSTOM_OBJECTS[name] = cls
                GLOBAL_CUSTOM_OBJECTS[f"Custom>{name}"] = cls
                GLOBAL_CUSTOM_OBJECTS[f"transformers>{name}"] = cls
                if hasattr(mod, "__name__"):
                    GLOBAL_CUSTOM_OBJECTS[f"{mod.__name__}>{name}"] = cls

                # Registrar también en el sistema nativo de Keras 3
                try:
                    keras.saving.register_keras_serializable(package="Custom", name=name)(cls)
                    keras.saving.register_keras_serializable(package="transformers", name=name)(cls)
                except Exception:
                    pass

build_custom_objects_registry()

# =========================================================
# PARCHE DE DESERIALIZACIÓN EXHAUSTIVA EN KERAS 3
# =========================================================
try:
    import keras.src.saving.serialization_lib as keras_saving
    _orig_retrieve = keras_saving._retrieve_class_or_fn

    def _robust_retrieve(name, registered_name, module, obj_type, full_config, custom_objects=None):
        merged = dict(GLOBAL_CUSTOM_OBJECTS)
        if custom_objects:
            merged.update(custom_objects)

        # 1. Intento primario con el recuperador original
        try:
            return _orig_retrieve(name, registered_name, module, obj_type, full_config, merged)
        except (TypeError, ValueError):
            pass

        # 2. Búsqueda directa por variantes de nombre
        clean_name = name.split(">")[-1].split(".")[-1] if name else ""
        clean_reg = registered_name.split(">")[-1].split(".")[-1] if registered_name else ""

        keys_to_check = [registered_name, name, clean_reg, clean_name]
        for k in keys_to_check:
            if k and k in merged:
                return merged[k]

        # 3. Intentar importar módulo dinámicamente si viene en la config
        if module:
            try:
                mod = importlib.import_module(module)
                for k in [clean_reg, clean_name]:
                    if k and hasattr(mod, k):
                        return getattr(mod, k)
            except Exception:
                pass

        # 4. Búsqueda exhaustiva en todos los módulos cargados en memoria
        for search_term in [clean_reg, clean_name]:
            if not search_term:
                continue
            for mod_name, mod in list(sys.modules.items()):
                if mod and ("transformers" in mod_name or "keras" in mod_name or "tensorflow" in mod_name):
                    if hasattr(mod, search_term):
                        obj = getattr(mod, search_term)
                        if isinstance(obj, type):
                            return obj

        err_msg = f"No se pudo localizar {obj_type} '{name}' (registered_name='{registered_name}', module='{module}')"
        print(f"❌ ERROR CRÍTICO DE DESERIALIZACIÓN: {err_msg}")
        raise TypeError(err_msg)

    keras_saving._retrieve_class_or_fn = _robust_retrieve
except Exception as e:
    print(f"Aviso parche deserializador: {e}")

# =========================================================
# CONFIGURACIÓN BÁSICA Y DESCARGA DE MODELO
# =========================================================

MODEL_PATH = "my_model.keras"
DRIVE_FILE_ID = "1onGKhJcSQamYZsdJZuC9OKixT5igAaRZ"

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        print("Descargando modelo (.keras)...")
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Modelo listo ✅")

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

# =========================================================
# CARGA DEL MODELO Y TOKENIZER
# =========================================================

def load_model():
    descargar_modelo()
    build_custom_objects_registry()

    # Intentar cargar con keras nativo
    try:
        return keras.models.load_model(
            MODEL_PATH,
            custom_objects=GLOBAL_CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False
        )
    except Exception as e1:
        print(f"Aviso carga keras nativo: {e1}. Intentando con tf.keras...")

    # Reintento con tf.keras
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=GLOBAL_CUSTOM_OBJECTS,
        compile=False
    )

def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# =========================================================
# PREPROCESADO
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
# INFERENCIA + FORMATO JSON
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

    try:
        raw_outputs = model(inputs, training=False)
    except Exception:
        inputs_dict = {
            "audio_melspec": melspec_batch,
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
        }
        raw_outputs = model(inputs_dict, training=False)

    if isinstance(raw_outputs, (list, tuple)):
        output_list = [v.numpy() if hasattr(v, "numpy") else np.array(v) for v in raw_outputs]
    elif isinstance(raw_outputs, dict):
        output_list = [v.numpy() if hasattr(v, "numpy") else np.array(v) for v in raw_outputs.values()]
    else:
        output_list = [raw_outputs.numpy()]

    # Estructura del modelo: [moods_fused, va_fused, moods_audio, moods_text, va_audio, va_text]
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