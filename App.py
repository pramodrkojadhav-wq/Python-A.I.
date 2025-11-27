
import os
from typing import Dict, List

import streamlit as st
from google import genai
from google.genai import types


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Gemini Multi-Persona Chat",
    page_icon="💬",
    layout="wide",
)

# --- Custom CSS for a cleaner look ---
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #111827, #020617);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .persona-card {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        background: #020617;
        border: 1px solid rgba(148, 163, 184, 0.4);
        margin-bottom: 0.5rem;
    }
    .persona-title {
        font-weight: 600;
        font-size: 0.95rem;
    }
    .persona-desc {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .model-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.4);
        margin-left: 0.25rem;
    }
    .stChatMessage {
        background: rgba(15, 23, 42, 0.85) !important;
        border-radius: 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Personas & models
# -----------------------------
PERSONAS: Dict[str, Dict[str, str]] = {
    "General Assistant 💬": {
        "emoji": "💬",
        "prompt": (
            "You are a helpful, friendly AI assistant. "
            "Answer clearly, stay on topic, and use simple language when possible."
        ),
    },
    "Code Mentor 👨‍💻": {
        "emoji": "👨‍💻",
        "prompt": (
            "You are a senior software engineer and coding mentor. "
            "Explain concepts step by step, show clean code snippets, and suggest best practices."
        ),
    },
    "Productive Coach 🎯": {
        "emoji": "🎯",
        "prompt": (
            "You are a productivity and focus coach. "
            "Give concise, practical advice, frameworks, and step-by-step plans."
        ),
    },
    "Creative Writer ✨": {
        "emoji": "✨",
        "prompt": (
            "You are a creative writing partner. "
            "Write vivid, engaging text and suggest story ideas and improvements with a warm tone."
        ),
    },
    "Concise Analyst 📊": {
        "emoji": "📊",
        "prompt": (
            "You are a concise analytical assistant. "
            "Prioritize clarity, structure, and bullet points. Avoid unnecessary fluff."
        ),
    },
}

AVAILABLE_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


# -----------------------------
# Helpers
# -----------------------------
def get_api_key() -> str:
    """Get API key from sidebar or environment."""
    sidebar_key = st.session_state.get("sidebar_api_key") or ""
    if sidebar_key.strip():
        return sidebar_key.strip()

    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def build_prompt(persona_prompt: str, history: List[Dict], user_input: str) -> str:
    """
    Build a single text prompt that includes:
    - Persona / system behavior
    - Conversation history
    - Latest user message
    This is sent as a single `contents` string to Gemini.
    """
    lines = []

    # Persona / system behavior
    lines.append(f"System: {persona_prompt}")
    lines.append("")  # blank line

    # Previous turns
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    # Current user turn
    lines.append(f"User: {user_input}")
    lines.append("Assistant:")

    return "\n".join(lines)


def get_conversation_key(model_name: str, persona_name: str) -> str:
    return f"{model_name}__{persona_name}"


# -----------------------------
# Session state structure
# -----------------------------
if "conversations" not in st.session_state:
    # { conv_key: [ {role: "user"/"assistant", content: str}, ... ] }
    st.session_state.conversations = {}

if "sidebar_api_key" not in st.session_state:
    st.session_state.sidebar_api_key = ""


# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("#### 🔑 Gemini API Key")
    st.session_state.sidebar_api_key = st.text_input(
        "Paste your Gemini API key",
        type="password",
        placeholder="Or set GEMINI_API_KEY / GOOGLE_API_KEY env var",
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("#### 🧠 Persona")
    persona_name = st.selectbox("Persona", list(PERSONAS.keys()))
    persona = PERSONAS[persona_name]

    st.markdown(
        f"""
        <div class="persona-card">
            <div class="persona-title">{persona['emoji']} {persona_name}</div>
            <div class="persona-desc">{persona['prompt']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧬 Model")
    model_name = st.selectbox("Gemini model", AVAILABLE_MODELS, index=0)
    st.caption(
        f"Using **{model_name}**. You can change this anytime; "
        "each persona+model combo keeps its own history."
    )

    st.markdown("#### 🎨 Generation settings")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max output tokens", 256, 2048, 1024, 128)

    if st.button("🧹 Clear current chat"):
        conv_key = get_conversation_key(model_name, persona_name)
        st.session_state.conversations[conv_key] = []
        st.success("Chat cleared for this persona + model.")


# -----------------------------
# Main layout – header
# -----------------------------
st.markdown(
    f"""
    <h1 style="margin-bottom: 0.25rem;">{persona['emoji']} Gemini Multi-Persona Chat</h1>
    <p style="color:#9ca3af; margin-bottom: 1.5rem;">
        Choose a persona & model in the sidebar, then start chatting.
        Each combination keeps its own conversation history.
    </p>
    """,
    unsafe_allow_html=True,
)

api_key = get_api_key()
if not api_key:
    st.warning(
        "Please provide your **Gemini API key** in the sidebar or via "
        "`GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variable."
    )

# Determine current conversation bucket
conv_key = get_conversation_key(model_name, persona_name)
if conv_key not in st.session_state.conversations:
    st.session_state.conversations[conv_key] = []

messages = st.session_state.conversations[conv_key]

# -----------------------------
# Show existing chat history
# -----------------------------
for msg in messages:
    if msg["role"] == "user":
        avatar = "🧑"
    else:
        avatar = persona["emoji"]

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# -----------------------------
# Chat input + Gemini call
# -----------------------------
user_input = None
if api_key:
    user_input = st.chat_input("Type your message and press Enter...")
else:
    st.chat_input(
        "Enter your Gemini API key in the sidebar to start.",
        disabled=True,
    )

if user_input:
    # 1) Show user message immediately
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 2) Call Gemini
    with st.chat_message("assistant", avatar=persona["emoji"]):
        placeholder = st.empty()
        try:
            client = genai.Client(api_key=api_key)

            prompt = build_prompt(
                persona_prompt=persona["prompt"],
                history=messages[:-1],  # previous messages only
                user_input=user_input,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            assistant_text = response.text or "(No response text returned.)"
            placeholder.markdown(assistant_text)

            # 3) Save assistant message to history
            messages.append({"role": "assistant", "content": assistant_text})

        except Exception as e:
            error_msg = f"⚠️ Error calling Gemini API: `{e}`"
            placeholder.markdown(error_msg)
            messages.append({"role": "assistant", "content": error_msg}) 
