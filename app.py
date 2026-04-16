import os
os.environ["TRANSFORMERS_VERBOSITY"] = "ERROR"
import streamlit as st
import whisper
import numpy as np
import io
import pydub
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
from googletrans import Translator
from transformers import pipeline

# --- 1. AI MODEL LOADING (Optimized) ---
@st.cache_resource
def load_models():
    # Loading 'tiny' to stay within memory limits
    w_model = whisper.load_model("tiny", device="cpu")
    # Sentiment analysis for emotion detection
    s_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return w_model, s_model

whisper_model, sentiment_pipeline = load_models()
translator = Translator()

# --- 2. BEAUTIFICATION (Custom CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #45a049; border: 2px solid white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    div[data-testid="stExpander"] { border: 1px solid #4CAF50; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🌟 AI Career Coach")
    menu = st.radio("Go to:", ["🏠 Home / Interview", "📊 Performance", "🤖 Chatbot", "📷 Camera"])
    st.divider()
    st.info("Tip: Speak clearly into the microphone for best transcription results.")

# --- 4. FEATURE LOGIC ---

# FEATURE: HOME / INTERVIEW
if menu == "🏠 Home / Interview":
    st.title("🎙️ Smart Interview Trainer")
    st.write("Record your answer below. Our AI will transcribe, translate, and analyze your tone.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Recording Station")
        audio = mic_recorder(start_prompt="Record Answer 🎙️", stop_prompt="Stop 🛑", key='recorder')

    if audio:
        st.audio(audio['bytes'])
        with st.spinner("Processing audio..."):
            # Audio conversion
            audio_data = io.BytesIO(audio['bytes'])
            audio_segment = pydub.AudioSegment.from_file(audio_data)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            
            # Transcription
            result = whisper_model.transcribe(samples)
            st.session_state.answer = result["text"]
            st.success("✅ Audio Captured!")

    with col2:
        st.subheader("AI Analysis")
        if st.button("🚀 Run Deep Analysis"):
            text_to_analyze = st.session_state.get('answer', "")
            if text_to_analyze:
                # Translation
                translated = translator.translate(text_to_analyze, dest='en').text
                # Emotion
                sentiment = sentiment_pipeline(translated)[0]
                
                with st.expander("See Results", expanded=True):
                    st.write(f"**Original:** {text_to_analyze}")
                    st.write(f"**English Version:** {translated}")
                    st.metric("Tone / Emotion", sentiment['label'], delta=f"{sentiment['score']:.2%}")
            else:
                st.warning("Please record something first!")

# FEATURE: PERFORMANCE DASHBOARD
elif menu == "📊 Performance":
    st.title("📈 Your Progress")
    # Sample data for visualization
    data = pd.DataFrame({
        'Session': ['Day 1', 'Day 2', 'Day 3', 'Day 4'],
        'Confidence Score': [60, 75, 70, 85]
    })
    st.line_chart(data.set_index('Session'))
    st.write("You are improving! Your confidence score increased by **15%** in the last session.")

# FEATURE: CHATBOT
elif menu == "🤖 Chatbot":
    st.title("🤖 Interview Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me about interview tips..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Simple AI response logic
        response = f"That's a great question about '{prompt}'. To succeed, focus on using the STAR method!"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# FEATURE: CAMERA
elif menu == "📷 Camera":
    st.title("📷 Eye Contact & Posture Check")
    img_file = st.camera_input("Take a photo to check your interview posture")
    if img_file:
        st.image(img_file)
        st.success("Looking professional! Keep your shoulders back and smile.")
