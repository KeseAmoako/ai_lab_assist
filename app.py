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

# Custom CSS for a clean, academic look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .safety-warning { 
        background-color: #ff4b4b22; 
        border: 1px solid #ff4b4b; 
        padding: 10px; 
        border-radius: 5px; 
        color: #ff4b4b;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are the "AI Lab Assistant," a patient, encouraging, and extremely safety-conscious expert for science students.
CORE RULES:
1. SAFETY FIRST: Before providing instructions, briefly remind the student of required PPE (goggles, gloves, lab coat) if the context involves chemicals or heat.
2. If a student describes a dangerous action (e.g., "smelling a chemical directly" or "pouring water into acid"), stop them immediately and explain the correct safety procedure (e.g., "wafting" or "Acid to Water").
3. FORMATTING: Use clear, numbered steps for protocols. Use bold text for equipment names.
4. TONE: Be supportive. If they are confused, explain concepts using simple analogies.
5. MULTIMODAL: If an image is provided, identify the equipment or chemical label clearly and explain its function.
"""

# --- 3. HELPER FUNCTIONS ---

def get_tts_audio(text):
    """Converts text to speech using Edge-TTS (high quality, free)."""
    # Filter out markdown symbols for cleaner speech
    clean_text = text.replace("*", "").replace("#", "").replace("- ", "")
    communicate = edge_tts.Communicate(clean_text, "en-US-GuyNeural")
    
    # Save to a byte stream
    audio_data = io.BytesIO()
    async def run_tts():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
    
    asyncio.run(run_tts())
    return audio_data.getvalue()

# --- 4. SIDEBAR & SESSION STATE ---
with st.sidebar:
    st.title("🧪 Lab Settings")
    api_key = st.text_input("Enter Google API Key:", type="password")
    st.info("Get a key at [aistudio.google.com](https://aistudio.google.com/)")
    
    voice_enabled = st.toggle("Voice Mode (AI speaks back)", value=True)
    
    if st.button("Clear Lab Notebook (History)"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. INITIALIZE AI MODEL ---
if api_key:
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash for speed and multimodal capabilities
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        system_instruction=SYSTEM_INSTRUCTION
    )
else:
    st.warning("Please enter your Google API Key in the sidebar to begin.")
    st.stop()

# --- 6. UI: CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], caption="Uploaded Image", width=300)

# --- 7. UI: INPUT AREA ---
st.markdown("---")
# Container for inputs to keep them together
with st.container():
    img_file = st.file_uploader("📸 Identify Equipment (Upload/Take Photo)", type=['png', 'jpg', 'jpeg'])
    audio_input = st.audio_input("🎤 Ask via Voice")
    chat_input = st.chat_input("Type your question here (e.g., 'How do I use a burette?')")

# --- 8. LOGIC: PROCESSING INPUTS ---
if chat_input or img_file or audio_input:
    
    # Initialize prompt components
    prompt_parts = []
    
    # 1. Handle Text
    user_content = chat_input if chat_input else "What am I looking at/listening to?"
    prompt_parts.append(user_content)
    
    # 2. Handle Image
    raw_image = None
    if img_file:
        raw_image = Image.open(img_file)
        prompt_parts.append(raw_image)
    
    # 3. Handle Audio Input (STT via Gemini's multimodal capability)
    if audio_input:
        # Gemini 1.5 can process audio bytes directly
        audio_bytes = audio_input.read()
        prompt_parts.append({"mime_type": "audio/wav", "data": audio_bytes})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_content)
        if raw_image:
            st.image(raw_image, width=300)
        if audio_input:
            st.audio(audio_input)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting Lab Manual..."):
            try:
                # Add history context (simplification: last 4 messages to save tokens)
                # In a production app, you'd format st.session_state.messages for the model
                response = model.generate_content(prompt_parts)
                response_text = response.text
                
                st.markdown(response_text)
                
                # Handle Voice Output
                if voice_enabled:
                    audio_output = get_tts_audio(response_text)
                    st.audio(audio_output, format="audio/mp3", autoplay=True)
                
                # Save to history
                history_entry = {"role": "assistant", "content": response_text}
                if raw_image:
                    history_entry["image"] = raw_image
                st.session_state.messages.append({"role": "user", "content": user_content})
                st.session_state.messages.append(history_entry)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Check if your API key is correct or if the file format is supported.")





