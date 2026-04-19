import streamlit as st
import random
import pandas as pd
import os

DB_FILE = "users_db.csv"

# Create the database file if it doesn't exist
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=["email", "password", "name"])
    df.to_csv(DB_FILE, index=False)

def save_user(email, password, name):
    df = pd.read_csv(DB_FILE)
    new_user = pd.DataFrame([[email, password, name]], columns=["email", "password", "name"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def verify_user(email, password):
    df = pd.read_csv(DB_FILE)
    user = df[(df['email'] == email) & (df['password'] == str(password))]
    return user if not user.empty else None
import matplotlib.pyplot as plt
import speech_recognition as sr
import cv2
import streamlit_authenticator as stauth
from googletrans import Translator
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import streamlit as st
import random
import time #

# --- 1. INITIALIZE SESSION STATE (Put it here!) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

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
    
# --- STEP 3: THE MULTI-PAGE UI ---

# 1. If not logged in, show Login/Signup tabs
if not st.session_state.logged_in:
    st.title("🔐 AI Interview Trainer")
    tab1, tab2 = st.tabs(["Login", "Create Account"])

    with tab2: # SIGN UP
        st.subheader("Create a New Account")
        new_name = st.text_input("Full Name")
        new_email = st.text_input("Email", key="signup_email")
        new_pw = st.text_input("Create Password", type="password")
        if st.button("Register"):
            if new_email and new_pw and new_name:
                save_user(new_email, new_pw, new_name)
                st.success("Account created! Now please go to the Login tab.")
            else:
                st.warning("Please fill in all fields.")

    with tab1: # LOGIN
        st.subheader("Login to your Account")
        email_input = st.text_input("Email", key="login_email")
        pw_input = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login"):
            user_data = verify_user(email_input, pw_input)
            if user_data is not None:
                # Get name from database instead of extraction
                st.session_state.user_display_name = user_data.iloc[0]['name']
                st.session_state.logged_in = True
                st.session_state.page = "welcome" 
                st.rerun()
            else:
                st.error("Invalid email or password.")

# 2. If logged in, handle the different pages (Welcome -> Role Selection)
# --- THIS GOES AFTER YOUR LOGIN/TAB CODE ---
else:
    # 1. This creates the 'menu' variable so the error disappears!
    with st.sidebar:
        st.write(f"👤 **Logged in as:** {st.session_state.user_display_name}")
        menu = st.selectbox("Menu", ["🏠 Home", "🎯 Interview"])
        st.divider()
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # 2. HOME PAGE
    if menu == "🏠 Home":
        st.title(f"✨ Welcome, {st.session_state.user_display_name}!")
        st.write("Your account is ready. Select 'Interview' from the sidebar to start.")

    # 3. INTERVIEW PAGE
    elif menu == "🎯 Interview":
        st.title("🎯 Interview Practice")
        
        # Combine Roles and Streams for the dropdown
        options = []
        for role, streams in ROLE_QUESTIONS.items():
            for stream in streams:
                options.append(f"{role} - {stream}")
        
        choice = st.selectbox("Pick your path:", options)
        role_part, stream_part = choice.split(" - ")
        questions = ROLE_QUESTIONS[role_part][stream_part]

        if st.button("Get New Question"):
            st.session_state.current_q = random.choice(questions)
            st.session_state.start_time = None

        if "current_q" in st.session_state:
            st.info(f"**Question:** {st.session_state.current_q}")
            
            # --- TIMER LOGIC ---
            if st.button("⏱️ Start 60s Timer"):
                st.session_state.start_time = time.time()

            if st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                remaining = max(0, 60 - int(elapsed))
                if remaining > 0:
                    st.metric("Time Remaining", f"{remaining}s")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("⏰ Time's Up!")
        # --- ROLE SELECTION LOGIC ---
        combined_options = []
        for role_name, streams in ROLE_QUESTIONS.items():
            for stream_name in streams.keys():
                combined_options.append(f"{role_name} - {stream_name}")

        selected_path = st.selectbox("🎯 Choose your Interview Path:", combined_options)
        role_choice, stream_choice = selected_path.split(" - ")
        current_list = ROLE_QUESTIONS[role_choice][stream_choice]

        if 'active_q' not in st.session_state or st.button("🔄 Change Question"):
            st.session_state.active_q = random.choice(current_list)
            st.session_state.start_time = None 

        st.info(f"**Interview Question:** {st.session_state.active_q}")

        # --- TIMER LOGIC ---
        timer_placeholder = st.empty()
        if st.button("⏱️ Start 60s Timer"):
            st.session_state.start_time = time.time()

        if st.session_state.get("start_time"):
            elapsed = time.time() - st.session_state.start_time
            remaining = max(0, 60 - int(elapsed))
            if remaining > 0:
                timer_placeholder.metric("Time Remaining", f"{remaining}s")
                time.sleep(1)
                st.rerun()
            else:
                timer_placeholder.error("⏰ Time's Up!")
                st.session_state.start_time = None
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
