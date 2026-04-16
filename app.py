import os
os.environ["TRANSFORMERS_VERBOSITY"] = "ERROR"
import streamlit as st
import whisper
import numpy as np
import io
import pydub
import pandas as pd
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth
from streamlit_mic_recorder import mic_recorder
from googletrans import Translator
from transformers import pipeline

# --- 1. AI MODEL LOADING ---
@st.cache_resource
def load_models():
    w_model = whisper.load_model("tiny", device="cpu")
    s_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return w_model, s_model

whisper_model, sentiment_pipeline = load_models()
translator = Translator()

# --- 2. HELPER FUNCTIONS ---
def to_english(text):
    try:
        return translator.translate(text, dest='en').text
    except:
        return text

def detect_emotion(text):
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        emoji = "😊 Positive/Confident" if label == "POSITIVE" else "😟 Negative/Stressed"
        return f"{emoji} ({score:.1%})"
    except:
        return "Unknown"

# --- 3. BEAUTIFICATION (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.02); border: 2px solid white; }
    div[data-testid="stMetricValue"] { color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION ---
with st.sidebar:
    st.title("🌟 Career Mentor AI")
    menu = st.radio("Navigation", ["🏠 Interview Prep", "📊 Performance", "🤖 AI Chatbot", "📷 Camera Check"])
    st.divider()
    st.info("Tip: After recording, click 'Analyze Now' to see your results.")

# --- 5. APP FEATURES ---

if menu == "🏠 Interview Prep":
    st.title("🎙️ Smart Interview Trainer")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: Record")
        audio = mic_recorder(start_prompt="Start Recording 🎙️", stop_prompt="Stop 🛑", key='recorder')

    if audio:
        st.audio(audio['bytes'])
        with st.spinner("Transcribing..."):
            audio_data = io.BytesIO(audio['bytes'])
            audio_segment = pydub.AudioSegment.from_file(audio_data)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            result = whisper_model.transcribe(samples)
            st.session_state.answer = result["text"]
        st.success("✅ Recorded!")

    with col2:
        st.subheader("Step 2: Analyze")
        if st.button("🚀 Analyze Now"):
            final_text = st.session_state.get('answer', "")
            if final_text:
                with st.spinner("Analyzing tone..."):
                    english_text = to_english(final_text)
                    emotion_result = detect_emotion(english_text)
                    st.metric("Tone Assessment", emotion_result)
                    with st.expander("Show Detailed Transcription"):
                        st.write(f"**English:** {english_text}")
            else:
                st.warning("Please record audio first.")

elif menu == "📊 Performance":
    st.title("📈 Performance Tracking")
    chart_data = pd.DataFrame({'Session': [1, 2, 3, 4], 'Score': [40, 65, 55, 80]})
    st.line_chart(chart_data.set_index('Session'))

elif menu == "🤖 AI Chatbot":
    st.title("🤖 Interview Coach Chat")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if prompt := st.chat_input("Ask for advice..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        reply = f"Focus on using the STAR method for that!"
        with st.chat_message("assistant"): st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

elif menu == "📷 Camera Check":
    st.title("📷 Posture & Eye Contact")
    photo = st.camera_input("Check your framing")
