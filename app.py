import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import os
from PIL import Image
import io

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="AI Lab Assistant", 
    page_icon="🥽", 
    layout="centered"
)

# Custom CSS for a clean look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. API KEY SECURITY ---
# This automatically reads the API key you saved in the Streamlit Cloud Dashboard Secrets
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔒 Developer Warning: Please add 'GOOGLE_API_KEY' to your Streamlit Secrets Dashboard.")
    st.stop()

# --- 3. SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are the "AI Lab Assistant," a patient, encouraging, and extremely safety-conscious expert for science students.
1. SAFETY FIRST: Remind students of PPE (goggles/gloves) if chemicals or heat are involved.
2. If multiple images are uploaded, identify EACH piece of equipment or label clearly.
3. Use bullet points for steps. Keep explanations simple for novices.
"""

# --- 4. SELF-HEALING MODEL INITIALIZATION ---
@st.cache_resource
def init_model():
    """Programmatically detects available models so the app never throws a 404 again."""
    genai.configure(api_key=API_KEY)
    try:
        # Get all models supported by your API key
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1. Try modern stable models first
        priority_models = [
            "models/gemini-2.5-flash",
            "models/gemini-3.5-flash",
            "models/gemini-1.5-flash"
        ]
        
        selected_model = None
        for model_name in priority_models:
            if model_name in available_models:
                selected_model = model_name
                break
        
        # 2. Dynamic fallback: pick any model that contains "flash" in the name
        if not selected_model:
            flash_models = [m for m in available_models if "flash" in m.lower()]
            if flash_models:
                selected_model = flash_models[0]
                
        # 3. Hard fallback if list fails or no matches found
        if not selected_model:
            selected_model = "models/gemini-2.5-flash"
            
        return genai.GenerativeModel(model_name=selected_model, system_instruction=SYSTEM_INSTRUCTION)
        
    except Exception:
        # Default safety fallback
        return genai.GenerativeModel(model_name="models/gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTION)

model = init_model()

# --- 5. AUDIO TTS HELPER ---
def get_tts_audio(text):
    """Converts the AI response into natural sounding speech."""
    clean_text = text.replace("*", "").replace("#", "").replace("- ", "")
    communicate = edge_tts.Communicate(clean_text, "en-US-GuyNeural")
    audio_data = io.BytesIO()
    async def run_tts():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
    asyncio.run(run_tts())
    return audio_data.getvalue()

# --- 6. GREETING & CHAT HISTORY ---
st.markdown("""
    <div style="text-align: center; padding: 10px 0px 30px 0px;">
        <h1 style="color: #2e7d32; font-size: 42px; margin-bottom: 5px;">🧪 Hello, Scientist!</h1>
        <p style="font-size: 18px; color: #555;">Ready to explore? Make sure to wear your safety goggles! 🥽</p>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            cols = st.columns(min(len(message["images"]), 4))
            for idx, img in enumerate(message["images"]):
                cols[idx % 4].image(img, use_container_width=True)

# Sidebar utilities
with st.sidebar:
    st.title("🥽 Lab Controller")
    voice_enabled = st.toggle("Voice Mode (AI speaks back)", value=True)
    if st.button("Clear Lab Notebook (History)"):
        st.session_state.messages = []
        st.rerun()

# --- 7. MULTI-INPUT CONTROLS ---
st.markdown("---")
with st.container():
    # File uploader with multiple image capability
    img_files = st.file_uploader(
        "📸 Upload Photos (Select multiple lab equipment pictures at once)", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )
    audio_input = st.audio_input("🎤 Describe your experiment or ask a question")
    chat_input = st.chat_input("Type your question here...")

# --- 8. LOGIC PROCESSING ---
if chat_input or img_files or audio_input:
    prompt_parts = []
    current_images = []
    
    # Text input
    user_text = chat_input if chat_input else "Please analyze the uploaded lab equipment."
    prompt_parts.append(user_text)
    
    # Process multiple images
    if img_files:
        for uploaded_file in img_files:
            img = Image.open(uploaded_file)
            current_images.append(img)
            prompt_parts.append(img)
            
    # Process voice input
    if audio_input:
        prompt_parts.append({"mime_type": "audio/wav", "data": audio_input.read()})

    # Show User's input immediately in UI
    with st.chat_message("user"):
        st.markdown(user_text)
        if current_images:
            cols = st.columns(min(len(current_images), 4))
            for idx, img in enumerate(current_images):
                cols[idx % 4].image(img, use_container_width=True)

    # Generate Assistant's reply
    with st.chat_message("assistant"):
        with st.spinner("Reviewing safety protocol..."):
            try:
                response = model.generate_content(prompt_parts)
                response_text = response.text
                
                st.markdown(response_text)
                
                # Speak if voice mode is on
                if voice_enabled:
                    st.audio(get_tts_audio(response_text), format="audio/mp3", autoplay=True)
                
                # Save to memory
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_text, 
                    "images": current_images
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")
