import streamlit as st
import google.generativeai as genai
import edge_tts
import asyncio
import os
from PIL import Image
import io
import uuid

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="AI Lab Assistant", page_icon="🥽", layout="wide")

# Custom CSS for Sidebar and Chat styling
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .chat-sidebar-item { padding: 10px; border-radius: 5px; margin-bottom: 5px; cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

# --- 2. API KEY SECURITY ---
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔒 Developer: Add 'GOOGLE_API_KEY' to Streamlit Secrets.")
    st.stop()

# --- 3. SESSION STATE INITIALIZATION ---
# We store multiple chats in a dictionary: { "Chat ID": { "name": "Lab 1", "messages": [] } }
if "chats" not in st.session_state:
    default_id = str(uuid.uuid4())
    st.session_state.chats = {
        default_id: {"name": "General Lab Help", "messages": []}
    }
    st.session_state.current_chat_id = default_id

# --- 4. MODEL INITIALIZATION ---
@st.cache_resource
def init_model():
    genai.configure(api_key=API_KEY)
    # Self-healing model selection
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest", # Or gemini-2.5-flash if available in your region
        system_instruction="You are a safety-first AI Lab Assistant. Use bullet points and prioritize PPE. Identify equipment in photos clearly."
    )

model = init_model()

# --- 5. AUDIO HELPER ---
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

# --- 6. SIDEBAR: CHAT MANAGEMENT ---
with st.sidebar:
    st.title("📓 Lab Notebooks")
    
    # Create New Chat
    new_chat_name = st.text_input("New Experiment Name:", placeholder="e.g. Titration Lab")
    if st.button("➕ Start New Chat", use_container_width=True):
        if new_chat_name:
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {"name": new_chat_name, "messages": []}
            st.session_state.current_chat_id = new_id
            st.rerun()

    st.divider()
    
    # List Existing Chats
    st.write("**Recent Experiments**")
    for chat_id, chat_data in list(st.session_state.chats.items()):
        col1, col2 = st.columns([0.8, 0.2])
        if col1.button(chat_data["name"], key=f"select_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()
        if col2.button("🗑️", key=f"del_{chat_id}"):
            if len(st.session_state.chats) > 1:
                del st.session_state.chats[chat_id]
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                st.rerun()
            else:
                st.warning("Keep at least one chat.")

    st.divider()
    voice_enabled = st.toggle("🔊 Voice Response", value=True)
    
    if st.button("⚠️ Clear All Notebooks", type="primary", use_container_width=True):
        st.session_state.chats = {str(uuid.uuid4()): {"name": "General Lab Help", "messages": []}}
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        st.rerun()

# --- 7. MAIN UI ---
current_chat = st.session_state.chats[st.session_state.current_chat_id]

st.markdown(f"### 🧪 Currently Assisting: **{current_chat['name']}**")

# Display Messages from Current Chat
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            cols = st.columns(min(len(message["images"]), 4))
            for idx, img in enumerate(message["images"]):
                cols[idx % 4].image(img, use_container_width=True)

# --- 8. INPUT AND LOGIC ---
st.markdown("---")
img_files = st.file_uploader("📸 Upload Equipment Photos", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
audio_input = st.audio_input("🎤 Ask via Voice")
chat_input = st.chat_input("Ask a lab question...")

if chat_input or img_files or audio_input:
    prompt_parts = []
    current_images = []
    user_text = chat_input if chat_input else "Analyze the uploaded files."
    
    # Gather context from previous messages in THIS chat (Memory)
    # We pass the last 5 messages for context
    history_context = current_chat["messages"][-5:]
    
    prompt_parts.append(user_text)
    if img_files:
        for f in img_files:
            img = Image.open(f)
            current_images.append(img)
            prompt_parts.append(img)
    if audio_input:
        prompt_parts.append({"mime_type": "audio/wav", "data": audio_input.read()})

    # UI Update
    with st.chat_message("user"):
        st.markdown(user_text)
        if current_images:
            cols = st.columns(min(len(current_images), 4))
            for idx, img in enumerate(current_images):
                cols[idx % 4].image(img, use_container_width=True)

    with st.chat_message("assistant"):
        try:
            response = model.generate_content(prompt_parts)
            st.markdown(response.text)
            
            if voice_enabled:
                st.audio(get_tts_audio(response.text), format="audio/mp3", autoplay=True)
            
            # Save to the specific chat in session state
            current_chat["messages"].append({"role": "user", "content": user_text, "images": current_images})
            current_chat["messages"].append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error: {e}")
