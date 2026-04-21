import json
import tempfile
import streamlit as st
import matplotlib.pyplot as plt

from inference import load_model, load_tokenizer, predict_and_format_song

st.set_page_config(page_title="Music Mood & Emotions", layout="centered")
st.title("🎵 Predicción de emociones y moods")
st.caption("Entradas: audio (mp3/wav/ogg) + letra (archivo .txt o texto). Salida: JSON.")

# =========================
# INTERPRETACIÓN AUTOMÁTICA
# =========================
def generate_emotional_interpretation(valence, arousal):
    if valence >= 0 and arousal >= 0:
        tone = "energetic and positive"
        extra = "suggesting excitement, happiness, or high engagement"
    elif valence < 0 and arousal >= 0:
        tone = "intense and tense"
        extra = "which may reflect stress, anger, or strong emotional charge"
    elif valence < 0 and arousal < 0:
        tone = "low-energy and negative"
        extra = "often associated with sadness, introspection, or melancholy"
    else:
        tone = "calm and positive"
        extra = "indicating relaxation, peacefulness, or emotional stability"

    intensity = abs(arousal)
    if intensity > 0.6:
        intensity_desc = "a strong emotional intensity"
    elif intensity > 0.3:
        intensity_desc = "a moderate emotional intensity"
    else:
        intensity_desc = "a soft and subtle emotional tone"

    return f"""
Overall, the song presents a **{tone}** profile, {extra}.  
It carries **{intensity_desc}**, which shapes how the emotion is perceived throughout the track.
"""

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
                title=title if title else "Not provided",
                artist= artist if artist else "Not provided",
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
    top_mood = moods[0]["mood"] if moods else "Emotion"
    quadrant = emotions["predicted_quadrant"]

    if quadrant == "Q1":
        quadrant_text = "This corresponds to a **high-energy positive emotional region**, often associated with feelings such as happiness, excitement, or joy."
    elif quadrant == "Q2":
        quadrant_text = "This corresponds to a **high-energy negative emotional region**, typically linked to emotions like anger, tension, or stress."
    elif quadrant == "Q3":
        quadrant_text = "This corresponds to a **low-energy negative emotional region**, commonly related to sadness, melancholy, or introspection."
    else:
        quadrant_text = "This corresponds to a **low-energy positive emotional region**, often reflecting calmness, relaxation, or peacefulness."

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
    - **Quadrant {emotions["predicted_quadrant"]}** → {quadrant_text}

    This helps to describe its overall emotional character.

    ⚠️ *Note: These values are model-based estimations and may vary depending on musical context and interpretation.*
    """)

    # 🔥 INTERPRETACIÓN AUTOMÁTICA
    interpretation = generate_emotional_interpretation(
        emotions["valence_normalized"],
        emotions["arousal_normalized"]
    )

    st.markdown("### 🧠 Model Interpretation")
    st.markdown(interpretation)

    st.divider()

    # =========================
    # GRÁFICO
    # =========================
    st.subheader("📊 Emotional Quadrant")

    valence = emotions["valence_normalized"]
    arousal = emotions["arousal_normalized"]

    fig, ax = plt.subplots()

    ax.axhline(0)
    ax.axvline(0)

    ax.scatter(valence, arousal, s=100)

    # Determinar cuadrante
    if valence >= 0 and arousal >= 0:
        color = "#FFD6E8"  # pastel rosado (Happy)
    elif valence < 0 and arousal >= 0:
        color = "#FFE5CC"  # pastel naranja (Angry)
    elif valence < 0 and arousal < 0:
        color = "#D6E0FF"  # pastel azul (Sad)
    else:
        color = "#D6F5E8"  # pastel verde (Calm)

    # Pintar cuadrante correspondiente
    if valence >= 0 and arousal >= 0:
        ax.fill_between([0, 1], 0, 1, alpha=0.4, color=color)
    elif valence < 0 and arousal >= 0:
        ax.fill_between([-1, 0], 0, 1, alpha=0.4, color=color)
    elif valence < 0 and arousal < 0:
        ax.fill_between([-1, 0], -1, 0, alpha=0.4, color=color)
    else:
        ax.fill_between([0, 1], -1, 0, alpha=0.4, color=color)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xlabel("Valence (Negative → Positive)")
    ax.set_ylabel("Arousal (Low → High)")
    ax.set_title("Emotion Position")

    ax.text(valence, arousal, f"  {top_mood}", verticalalignment='bottom')

    # Labels cuadrantes
    ax.text(0.5, 0.8, "Q1\nHappy / Excited")
    ax.text(-0.8, 0.8, "Q2\nAngry / Stressed")
    ax.text(-0.8, -0.8, "Q3\nSad / Depressed")
    ax.text(0.5, -0.8, "Q4\nCalm / Relaxed")

    st.pyplot(fig)

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

    # =========================
    # JSON OCULTO
    # =========================
    with st.expander("📄 View JSON output"):
        st.code(result_json_str, language="json")
        st.json(data)

    st.download_button(
    "Descargar JSON",
    data=result_json_str,
    file_name="prediccion.json",
    mime="application/json",
)

