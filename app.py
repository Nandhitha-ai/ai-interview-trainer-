import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
import cv2
import streamlit_authenticator as stauth
from googletrans import Translator
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Interview Trainer",
    page_icon="🎤",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body { background-color: #0f172a; }
.main { color: white; }
h1, h2, h3 { color: #38bdf8; text-align: center; }
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ---------------
# -------- SIMPLE LOGIN --------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

# --- Starting at Line 44 ---
    if st.button("Login"):
    # Everything below is indented by 4 spaces
        if username == "user1" and password == "1234":
           st.session_state.logged_in = True
           st.success("Login successful")
           st.rerun()
        else:
           st.error("Invalid username or password")
    st.stop()#this prevents the rest of the app from loading until login    
        
# ---------------- SIDEBAR ----------------
st.sidebar.title("🎤 AI Trainer")
menu = st.sidebar.radio("Navigation",
                        ["🏠 Home", "📊 Performance", "🤖 Chatbot", "📷 Camera"])

# ---------------- LANGUAGE ----------------
translator = Translator()
language = st.selectbox("Language", ["English", "Tamil"])

def to_english(text):
    if language == "Tamil":
        return translator.translate(text, dest='en').text
    return text

def to_tamil(text):
    if language == "Tamil":
        return translator.translate(text, dest='ta').text
    return text

# ---------------- QUESTIONS ----------------
questions = [
    "Tell me about yourself",
    "Why should we hire you?",
    "What are your strengths?",
    "Describe a challenge you faced"
]

# ---------------- AI MODELS ----------------
emotion_model = pipeline("sentiment-analysis",model="distilbert-base-uncased")

# ---------------- FUNCTIONS ----------------

def detect_emotion(text):
    result = emotion_model(text)[0]['label']
    return "Confident 😊" if result == "POSITIVE" else "Nervous 😟"

def calculate_score(text):
    words = text.split()
    length_score = min(len(words), 50)
    hesitation_words = ["um", "uh", "like"]
    hesitation_count = sum(word.lower() in hesitation_words for word in words)
    return max(length_score - hesitation_count * 2, 0)

def save_data(q, a, e, s):
    df = pd.DataFrame([[q, a, e, s]],
                      columns=["Question", "Answer", "Emotion", "Score"])
    try:
        old = pd.read_csv("data.csv")
        df = pd.concat([old, df])
    except:
        pass
    df.to_csv("data.csv", index=False)

def show_graph():
    try:
        data = pd.read_csv("data.csv")
        plt.figure()
        plt.plot(data["Score"])
        plt.xlabel("Attempts")
        plt.ylabel("Score")
        plt.title("Performance")
        st.pyplot(plt)
    except:
        st.warning("No data yet")

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return "Could not understand"

def chatbot_reply(text):
    prompt = "You are a professional interviewer.\nUser: " + text
    response = chatbot(prompt, max_length=100)
    return response[0]['generated_text']

def start_camera():
    cap = cv2.VideoCapture(0)
    st.info("Press Q to exit")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.title("🎤 AI Interview Trainer")
    question = random.choice(questions)
    st.info(question)

    answer = st.text_area("Your Answer", height=150)
    col1, col2 = st.columns(2)

    with col1:
        # REPLACE THE OLD BUTTON WITH THIS:
        audio = mic_recorder(
            start_prompt="Record Answer 🎙️",
            stop_prompt="Stop 🛑",
            key='recorder'
        )

        if audio:
            st.audio(audio['bytes'])
            # Save the audio bytes to session state so 'Analyze' can see it
            st.session_state.audio_data = audio['bytes']
            st.write("Audio recorded successfully!")

    with col2:
        analyze = st.button("🚀 Analyze")
   if analyze:
    # 1. Check if there is text or audio to analyze
    if answer or 'audio_data' in st.session_state:
        with st.spinner("Analyzing your response..."):
            # If audio exists, you might need to transcribe it first
            # processed = to_english(transcribed_text) 
            
            # 2. Run your AI analysis models
            processed = to_english(answer)
            emotion = detect_emotion(processed)
            score = calculate_score(processed)

            # 3. Display the results
            st.markdown("### 📊 Result")
            st.success(f"Emotion: {to_tamil(emotion)}")
            st.info(f"Score: {score}")

            # 4. Give feedback based on the score
            if score < 20:
                st.warning("Improve your answer")
            else:
                st.success("Good job!")
                
            # Save the data
            save_data(question, answer, emotion, score)
    else:
        st.error("Please provide an answer or record audio first!")
# ---------------- PERFORMANCE ----------------
elif menu == "📊 Performance":
    st.title("📈 Performance Dashboard")
    show_graph()

# ---------------- CHATBOT ----------------
elif menu == "🤖 Chatbot":
    st.title("🤖 AI Interviewer")

    user_input = st.text_input("Ask something")

    if st.button("Send"):
        reply = chatbot_reply(user_input)
        st.write("👔 Interviewer:", reply)

# ---------------- CAMERA ----------------
elif menu == "📷 Camera":
    st.title("📷 Face Detection")

    if st.button("Start Camera"):
        start_camera()
