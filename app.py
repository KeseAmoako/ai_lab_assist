import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import os
from PIL import Image
import io

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="AI Lab Assistant", page_icon="🥽", layout="centered")

# --- 2. API KEY SECURITY ---
# This looks for a Secret called "GOOGLE_API_KEY" in your Streamlit Dashboard
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Developer: Please add 'GOOGLE_API_KEY' to your Streamlit Secrets.")
    st.stop()

# --- 3. SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are the "AI Lab Assistant," a patient, encouraging, and extremely safety-conscious expert for science students.
1. SAFETY FIRST: Remind students of PPE (goggles/gloves) if chemicals or heat are involved.
2. If multiple images are uploaded, identify EACH piece of equipment or label clearly.
3. Use bullet points for steps. Keep explanations simple for novices.
"""

# --- 4. MODEL INITIALIZATION (UNIVERSAL FINDER) ---
@st.cache_resource
def init_model():
    genai.configure(api_key=API_KEY)
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    # Order of preference
    selected_name = "models/gemini-1.5-flash-latest" # Default
    for pref in ["models/gemini-1.5-flash-latest", "models/gemini-1.5-flash", "models/gemini-pro"]:
        if pref in available_models:
            selected_name = pref
            break
            
    return genai.GenerativeModel(model_name=selected_name, system_instruction=SYSTEM_INSTRUCTION)

model = init_model()

# --- 5. HELPER FUNCTIONS ---
def get_tts_audio(text):
    clean_text = text.replace("*", "").replace("#", "").replace("- ", "")
    communicate = edge_tts.Communicate(clean_text, "en-US-GuyNeural")
    audio_data = io.BytesIO()
    async def run_tts():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
    asyncio.run(run_tts())
    return audio_data.getvalue()

# --- 6. UI & SESSION STATE ---

st.title("👋 Welcome to the AI Lab Assistant!")
st.subheader("Your guide to safe and successful experiments.")
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #007bff; font-size: 50px;">🧪 Hello, Scientist!</h1>
        <p style="font-size: 20px; color: #555;">Ready to explore? Don't forget your goggles! 🥽</p>
    </div>
    """, unsafe_allow_html=True)

# Optional: A nice status badge
st.info("💡 Tip: You can upload photos of equipment or speak directly to me for help.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🧪 Lab Assistant")
    voice_enabled = st.toggle("Voice Mode (AI speaks)", value=True)
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            cols = st.columns(len(message["images"]))
            for idx, img in enumerate(message["images"]):
                cols[idx].image(img, width=150)

# --- 7. MULTI-INPUT AREA ---
st.markdown("---")
with st.container():
    # ALLOW MULTIPLE FILES
    img_files = st.file_uploader("📸 Upload Lab Photos", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
    audio_input = st.audio_input("🎤 Speak to Assistant")
    chat_input = st.chat_input("What's on ya mind...")

# --- 8. LOGIC ---
if chat_input or img_files or audio_input:
    prompt_parts = []
    current_images = []
    
    # Process text
    user_text = chat_input if chat_input else "Please analyze the uploaded content."
    prompt_parts.append(user_text)
    
    # Process multiple images
    if img_files:
        for uploaded_file in img_files:
            img = Image.open(uploaded_file)
            current_images.append(img)
            prompt_parts.append(img)
            
    # Process audio
    if audio_input:
        prompt_parts.append({"mime_type": "audio/wav", "data": audio_input.read()})

    # Show user input
    with st.chat_message("user"):
        st.markdown(user_text)
        if current_images:
            cols = st.columns(len(current_images))
            for idx, img in enumerate(current_images):
                cols[idx].image(img, width=150)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = model.generate_content(prompt_parts)
                st.markdown(response.text)
                
                if voice_enabled:
                    st.audio(get_tts_audio(response.text), format="audio/mp3", autoplay=True)
                
                # Save to history
                st.session_state.messages.append({"role": "user", "content": user_text, "images": current_images})
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {e}")
