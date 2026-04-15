import json
import tempfile
import streamlit as st

from inference import load_model, load_tokenizer, predict_and_format_song

st.set_page_config(page_title="Music Mood & Emotions", layout="centered")
st.title("🎵 Predicción de emociones y moods")
st.caption("Entradas: audio (mp3/wav/ogg) + letra (archivo .txt o texto). Salida: JSON.")

@st.cache_resource
def get_model_and_tokenizer():
    model = load_model()
    tok = load_tokenizer()
    return model, tok

model, tokenizer = get_model_and_tokenizer()

audio_file = st.file_uploader("Sube el audio", type=["mp3", "wav", "ogg"])

lyrics_mode = st.radio(
    "Letra: ¿cómo la vas a ingresar?",
    ["Desde archivo .txt", "Escribir/pegar texto"],
    horizontal=True
)

lyrics_file = None
lyrics_text = None

if lyrics_mode == "Desde archivo .txt":
    lyrics_file = st.file_uploader("Sube la letra (.txt)", type=["txt"])
else:
    lyrics_text = st.text_area("Pega la letra aquí", height=220)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Título (opcional)")
with col2:
    artist = st.text_input("Artista (opcional)")

mood_threshold = st.slider("Umbral para incluir mood", 0.0, 0.9, 0.3, 0.05)

if st.button("🔍 Analizar"):
    if audio_file is None:
        st.error("Sube un archivo de audio primero.")
        st.stop()

    if lyrics_mode == "Desde archivo .txt" and lyrics_file is None:
        st.error("Sube un archivo .txt con la letra.")
        st.stop()

    if lyrics_mode == "Escribir/pegar texto" and (not lyrics_text or not lyrics_text.strip()):
        st.error("Debes pegar o escribir la letra.")
        st.stop()

    with st.spinner("Procesando..."):
        # Guardar audio temporal
        suffix = "." + audio_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
            tmp_audio.write(audio_file.read())
            audio_path = tmp_audio.name

        lyric_path = None
        lyric_text_value = None

        # Guardar letra temporal si viene por archivo
        if lyrics_mode == "Desde archivo .txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(lyrics_file.read())
                lyric_path = tmp_txt.name
        else:
            lyric_text_value = lyrics_text

        try:
            result_json_str = predict_and_format_song(
                model=model,
                tokenizer=tokenizer,
                audio_file_path=audio_path,
                lyric_file_path=lyric_path,
                lyric_text=lyric_text_value,
                title=title if title else None,
                artist=artist if artist else None,
                mood_threshold=mood_threshold,
            )
        except Exception as e:
            st.error(f"Error en inferencia: {e}")
            st.stop()

    st.success("Listo ✅")

    data = json.loads(result_json_str)

    song = data["song_info"]
    emotions = data["emotions"]
    moods = data["moods_classification"]

    st.subheader("🎶 Song Analysis")

    st.markdown(f"""
    **Title:** {song["title"]}  
    **Artist:** {song["artist"]}  
    """)

    st.divider()

    # =========================
    # EMOTIONAL PROFILE
    # =========================
    st.subheader("💡 Emotional Interpretation")

    st.markdown(f"""
    Based on both the **audio characteristics** and the **lyrical content**, the model suggests that this song is most closely associated with:

    ### 👉 *{emotions["description"]}*

    This interpretation comes from two core dimensions:

    - **Valence ({emotions["valence_normalized"]})** → reflects how *positive or negative* the emotional tone is.
    - **Arousal ({emotions["arousal_normalized"]})** → indicates the level of *energy or intensity*.

    Together, these place the song in **Quadrant {emotions["predicted_quadrant"]}**, which helps describe its overall emotional character.

    ⚠️ *Note: These values are model-based estimations and may vary depending on musical context and interpretation.*
    """)

    st.divider()

    # =========================
    # MOODS
    # =========================
    st.subheader("🎭 Mood Breakdown")

    if moods:
        st.markdown("""
    The following moods were identified as the most relevant for this song.
    Percentages represent their relative contribution among detected moods:
    """)

        for mood in moods:
            st.markdown(f"- **{mood['mood']}** → {mood['percentage']}%")

        st.markdown("""
    💡 *These moods are not exclusive — a song can express multiple emotional tones at the same time.*
    """)

    else:
        st.info("No dominant moods were detected above the selected threshold.")

        data = json.loads(result_json_str)

    with st.expander("📄 View JSON output"):
        st.code(result_json_str, language="json")
        st.json(data)

    st.download_button(
    "Descargar JSON",
    data=result_json_str,
    file_name="prediccion.json",
    mime="application/json",
)

