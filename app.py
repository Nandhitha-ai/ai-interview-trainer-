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
import streamlit as st
import random
import time

# --- 1. INITIALIZE SESSION STATE (Put it here!) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

# --- 2. YOUR DICTIONARY ---
ROLE_QUESTIONS = {
    "Software Developer": {
        "Python/Backend": ["..."],
        # ... the rest of your questions
import streamlit as st
import random

# --- PART 1: THE DATA ---
# This is just a dictionary. It doesn't show up on screen yet.
ROLE_QUESTIONS = {
    "Software Developer": {
        "Python/Backend": [
            "Explain the difference between a list and a tuple.",
            "What are decorators in Python?"
        ],
        "Web Frontend": [
            "How would you optimize a Streamlit app?",
            "What is state management?"
        ]
    },
    "Data Analyst": {
        "Statistics": [
            "What is a P-value?",
            "Explain the normal distribution."
        ]
    }
}

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
# 1. Initialize session state to track login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_display_name" not in st.session_state:
    st.session_state.user_display_name = ""

# 2. LOGIN PAGE UI
if not st.session_state.logged_in:
    st.title("🔐 AI Interview Trainer Login")
    
    with st.container():
        email = st.text_input("Email Address", placeholder="name@example.com")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            # Check credentials (replace with your database logic if needed)
            if email == "user@gmail.com" and password == "1234":
                st.session_state.logged_in = True
                
                # --- NEW FEATURE: Name Extraction ---
                # Example: suresh_kumar@gmail.com -> Suresh Kumar
                display_name = email.split('@')[0].replace('_', ' ').replace('.', ' ').title()
                st.session_state.user_display_name = display_name
                
                st.success(f"Welcome back, {display_name}!")
                st.rerun()
            else:
                st.error("Invalid email or password. Please try again.")
    
    # This stops the rest of the app from loading until the user logs in
    st.stop()

# 3. LOGOUT FEATURE (In the Sidebar)
with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state.user_display_name}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_display_name = ""
        st.rerun()

# 4. MAIN APP WELCOME MESSAGE
st.title(f"👋 Welcome, {st.session_state.user_display_name}!")
st.write("Pick a category to start your interview practice.")
st.divider()
# 1. Create a combined list of all roles and streams
# This creates a list like: ["Software Developer - Python/Backend", "Data Analyst - Statistics"]
combined_options = []
for role, streams in ROLE_QUESTIONS.items():
    for stream in streams.keys():
        combined_options.append(f"{role} - {stream}")

# 2. Show only ONE selectbox
selected_path = st.selectbox("🎯 Choose your Interview Path:", combined_options)

# 3. Split the choice back into Role and Stream to get the questions
# If they pick "Data Analyst - Statistics", this splits it back apart
role_choice, stream_choice = selected_path.split(" - ")
current_list = ROLE_QUESTIONS[role_choice][stream_choice]

# 4. Pick and show the question
if 'active_q' not in st.session_state or st.button("🔄 Change Question"):
    st.session_state.active_q = random.choice(current_list)

st.info(f"**Interview Question:** {st.session_state.active_q}")
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
    return "Confident 😊" if result.upper() == "POSITIVE" else "Nervous 😟"

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
          if answer or 'audio_data' in st.session_state:
            with st.spinner("Analyzing your response..."):
                # All these lines must be indented inside the spinner block
                processed = to_english(answer)
                emotion = detect_emotion(processed)
                score = calculate_score(processed)
                with st.expander("See Detailed Analysis"):
                    st.write(f"Refined Answer: {processed}")
                    st.write(f"Detected Emotion: {to_tamil(emotion)}")

                    st.markdown("### 📊 Result")
                    st.success(f"Emotion: {to_tamil(emotion)}")
                    st.info(f"Score: {score}")

                if score < 20:
                    st.warning("Improve your answer")
                else:
                    st.success("Good job!")
                
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
