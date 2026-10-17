import os

# Configuración de entorno para Transformers
os.environ["USE_TF"] = "1"

import json
import tempfile
import importlib
import inspect
import numpy as np
import librosa

import keras
import tensorflow as tf
from transformers import AutoTokenizer
import gdown

# =========================================================
# PARCHE RESILIENTE DE INVOCACIÓN DE CAPAS (CLASE BASE)
# =========================================================

def safe_invoke_layer(self_obj, call_fn, *args, **kwargs):
    clean_kw = dict(kwargs)
    clean_kw.pop("mask", None)

    args_list = list(args)

    # 1. Extraer o desempaquetar tensores de entrada si vienen en una lista/tupla
    if len(args_list) == 1 and isinstance(args_list[0], (list, tuple)):
        inner_inputs = args_list[0]
        
        # Probar desempacar posicionalmente: self.call(input_1, input_2, ...)
        try:
            return call_fn(self_obj, *inner_inputs)
        except TypeError:
            pass

        # Probar desempacar posicionalmente limpiando kwargs duplicados
        kw = dict(clean_kw)
        if len(inner_inputs) >= 2:
            kw.pop("attention_mask", None)
            kw.pop("token_type_ids", None)
            kw.pop("position_ids", None)

        try:
            return call_fn(self_obj, *inner_inputs, **kw)
        except TypeError:
            pass

        # Probar pasando la lista como primer argumento con kwargs limpios
        try:
            return call_fn(self_obj, inner_inputs, **kw)
        except TypeError:
            pass

    # 2. Probar llamada estándar con kwargs limpios
    try:
        return call_fn(self_obj, *args_list, **clean_kw)
    except TypeError:
        pass

    # 3. Intentar invocación directa de .call()
    if hasattr(self_obj, "call"):
        if len(args_list) == 1 and isinstance(args_list[0], (list, tuple)):
            try:
                return self_obj.call(*args_list[0])
            except TypeError:
                pass
        try:
            return self_obj.call(*args_list, **clean_kw)
        except TypeError:
            pass

    raise TypeError(f"No se pudo invocar la capa {type(self_obj).__name__} de manera segura.")


def apply_class_level_layer_patches():
    # Obtener todas las clases base Layer de Keras y TF-Keras
    layer_classes = []

    for mod_name, class_name in [
        ("tf_keras.src.engine.base_layer", "Layer"),
        ("keras.src.engine.base_layer", "Layer"),
        ("keras.src.layers", "Layer"),
        ("keras.layers", "Layer"),
        ("tensorflow.keras.layers", "Layer"),
        ("transformers.modeling_tf_utils", "TFPreTrainedModel"),
        ("transformers.models.roberta.modeling_tf_roberta", "TFRobertaPreTrainedModel"),
        ("transformers.models.roberta.modeling_tf_roberta", "TFRobertaMainLayer"),
    ]:
        try:
            m = importlib.import_module(mod_name)
            cls = getattr(m, class_name, None)
            if cls and cls not in layer_classes:
                layer_classes.append(cls)
        except Exception:
            pass

    for cls in layer_classes:
        if hasattr(cls, "__call__"):
            orig_call = getattr(cls, "__call__")
            if not getattr(orig_call, "_is_patched", False):
                def make_patched_call(original_fn):
                    def patched_call(self, *args, **kwargs):
                        try:
                            out = original_fn(self, *args, **kwargs)
                        except TypeError:
                            out = safe_invoke_layer(self, getattr(self, "call", original_fn), *args, **kwargs)

                        if hasattr(out, "to_tuple"):
                            return out.to_tuple()
                        elif isinstance(out, dict) and not isinstance(out, tf.Tensor):
                            return tuple(out.values())
                        return out

                    patched_call._is_patched = True
                    return patched_call

                setattr(cls, "__call__", make_patched_call(orig_call))

# Aplicar parche a nivel de clase al importar el archivo
apply_class_level_layer_patches()

# =========================================================
# PARCHE RESILIENTE DE DESERIALIZACIÓN EN KERAS
# =========================================================
try:
    from keras.src.saving import serialization_lib
    _original_retrieve_class_or_fn = serialization_lib._retrieve_class_or_fn

    def _robust_retrieve_class_or_fn(name, registered_name, module, obj_type, full_config, custom_objects=None):
        try:
            return _original_retrieve_class_or_fn(name, registered_name, module, obj_type, full_config, custom_objects)
        except (TypeError, ValueError):
            pass

        keys_to_check = [
            registered_name,
            name,
            registered_name.split(">")[-1] if registered_name else None,
            name.split(">")[-1] if name else None,
            registered_name.split(".")[-1] if registered_name else None,
            name.split(".")[-1] if name else None,
        ]
        keys_to_check = [k for k in keys_to_check if k]

        if custom_objects:
            for k in keys_to_check:
                if k in custom_objects:
                    return custom_objects[k]

        if module:
            try:
                mod = importlib.import_module(module)
                for k in keys_to_check:
                    if hasattr(mod, k):
                        return getattr(mod, k)
            except Exception:
                pass

        for fallback_mod_name in ["transformers", "transformers.models.roberta.modeling_tf_roberta"]:
            try:
                mod = importlib.import_module(fallback_mod_name)
                for k in keys_to_check:
                    if hasattr(mod, k):
                        return getattr(mod, k)
            except Exception:
                pass

        raise TypeError(f"No se pudo localizar {obj_type} '{name}' (registered_name='{registered_name}')")

    serialization_lib._retrieve_class_or_fn = _robust_retrieve_class_or_fn
except Exception as e:
    print(f"Aviso en parche de deserialización: {e}")

# ==========================
# CONFIGURACIÓN BÁSICA Y DESCARGA
# ==========================

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

# ==========================
# CARGA DEL MODELO Y TOKENIZER
# ==========================

def load_model():
    descargar_modelo()
    apply_class_level_layer_patches()
    
    import transformers
    try:
        import transformers.models.roberta.modeling_tf_roberta as roberta_mod
    except Exception:
        roberta_mod = None

    custom_objects = {}
    modules_to_scan = [m for m in [transformers, roberta_mod] if m is not None]

    for mod in modules_to_scan:
        for name, cls in getattr(mod, "__dict__", {}).items():
            if isinstance(cls, type):
                custom_objects[name] = cls
                custom_objects[f"Custom>{name}"] = cls
                custom_objects[f"transformers>{name}"] = cls
                custom_objects[f"{mod.__name__}>{name}"] = cls

    try:
        return tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects=custom_objects, 
            compile=False
        )
    except Exception as e1:
        print(f"Aviso en carga con tf.keras: {e1}. Intentando con keras nativo...")
        try:
            scope = keras.utils.custom_object_scope(custom_objects)
        except AttributeError:
            scope = keras.saving.custom_object_scope(custom_objects)

        with scope:
            return keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                safe_mode=False
            )

def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# ==========================
# PREPROCESADO
# ==========================

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

    # 1) Preprocesar entradas
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

    melspec_batch = tf.convert_to_tensor(melspec_batch, dtype=tf.float32)
    input_ids_batch = tf.convert_to_tensor(input_ids_batch, dtype=tf.int32)
    attention_mask_batch = tf.convert_to_tensor(attention_mask_batch, dtype=tf.int32)

    # 2) Inferencia
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

    if isinstance(raw_outputs, dict):
        output_list = [v.numpy() for v in raw_outputs.values()]
    elif isinstance(raw_outputs, (list, tuple)):
        output_list = [v.numpy() for v in raw_outputs]
    else:
        output_list = [raw_outputs.numpy()]

    moods_fused_pred = None
    va_fused_pred = None

    for arr in output_list:
        if arr.ndim == 2 and arr.shape[-1] == 2:
            va_fused_pred = arr[0]
        elif arr.ndim == 2 and arr.shape[-1] == len(MOOD_LABELS):
            moods_fused_pred = arr[0]

    if moods_fused_pred is None or va_fused_pred is None:
        if len(output_list) >= 2:
            moods_fused_pred = output_list[0][0]
            va_fused_pred = output_list[1][0]
        else:
            raise ValueError(f"No se pudieron extraer las salidas requeridas del modelo. Salidas: {len(output_list)}")

    # 3) Emociones (Valence/Arousal)
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

    # 4) Moods
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

    # 5) JSON final
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