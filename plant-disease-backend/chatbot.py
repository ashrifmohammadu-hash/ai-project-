import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
CHAT_MODEL = "llama-3.3-70b-versatile"

# Lazily initialised on first use so Flask doesn't crash if API key is missing
_client = None


def _get_groq_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    _client = Groq(api_key=api_key)
    return _client

# ── Local knowledge base lookup ──

# Import disease info lazily to avoid circular import at module level
_DISEASE_INFO = None
_LOOKUP_TABLE = None


def _ensure_disease_info():
    global _DISEASE_INFO, _LOOKUP_TABLE
    if _DISEASE_INFO is not None:
        return
    from predict import DISEASE_INFO
    _DISEASE_INFO = DISEASE_INFO
    _LOOKUP_TABLE = {}
    for key, info in DISEASE_INFO.items():
        # Normalize the key: replace underscores/hyphens with spaces, remove brackets content
        norm = re.sub(r"\(.*?\)", "", key).replace("_", " ").replace("-", " ").lower().strip()
        norm = re.sub(r"\s+", " ", norm)
        _LOOKUP_TABLE[norm] = (key, info)
        # Also add the raw key as-is (for exact matches)
        plain = key.lower().replace("_", " ")
        _LOOKUP_TABLE[plain] = (key, info)


def _find_matching_disease(message):
    """Check if the message mentions a known disease name."""
    _ensure_disease_info()
    msg_lower = message.lower()
    # Sort by length descending so longer (more specific) names match first
    for normalized_name, (key, info) in sorted(
        _LOOKUP_TABLE.items(), key=lambda x: -len(x[0])
    ):
        if normalized_name in msg_lower:
            return (key, info)
    return None


def _format_treatment_reply(disease_key, info):
    """Format treatment info into a readable reply string."""
    lines = [f"**{disease_key.replace('_', ' ')}**\n"]
    if info.get("cause"):
        lines.append(f"*Cause:* {info['cause']}")
    if info.get("symptoms"):
        lines.append(f"*Symptoms:* {info['symptoms']}")
    if info.get("treatment"):
        lines.append(f"*Treatment:* {info['treatment']}")
    if info.get("prevention"):
        lines.append(f"*Prevention:* {info['prevention']}")
    if info.get("fertilizer"):
        lines.append(f"*Fertilizer:* {info['fertilizer']}")
    if info.get("weather"):
        lines.append(f"*Weather:* {info['weather']}")
    if info.get("insects"):
        lines.append(f"*Insects:* {info['insects']}")
    if info.get("severity"):
        lines.append(f"*Severity:* {info['severity']}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are Plant Village AI Assistant, a helpful chatbot for farmers and gardeners.

## Your role
- Answer questions about plant diseases, crop care, and agriculture.
- Give concise, practical advice suitable for farmers.
- Use simple language — avoid overly technical jargon.
- If you don't know something, say so honestly.

## About Plant Village AI
- The app lets users upload a leaf photo to get AI-powered disease diagnosis.
- The system uses a Groq vision model (Llama 4 Scout) to identify the plant and disease.
- Users can optionally fill a farmer questionnaire (plant type, symptoms, weather, insect damage, leaf age, part affected) to improve accuracy.
- The diagnosis returns: plant type, disease name, confidence score (0-100%), treatment info, symptoms, cause, prevention, and fertilizer advice.
- If confidence is low, the system asks clarification questions.
- Supported plants: Mango, Banana, Apple, Coconut, Rice, Tea, Chili, Papaya, Tomato, Potato, Corn, Grape.

## Supported diseases in knowledge base:
- Mango: Anthracnose, Bacterial canker, Die back, Powdery mildew, Sooty mould, healthy
- Banana: Sigatoka, Xanthomonas wilt, healthy
- Apple: Cedar apple rust, Apple scab, Black rot, healthy
- Coconut: Gray leaf spot, Leaf rot, healthy
- Rice: Blast, Bacterial blight, Brown spot, Tungro, healthy
- Tea: Algal leaf spot, Anthracnose, Bird eye spot, Brown blight, Gray blight, Red rust, White spot, healthy
- Chili: Bacterial spot, healthy
- Papaya: Anthracnose, Bacterial spot, Leaf curl, Ringspot, healthy
- Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, healthy
- Potato: Early blight, Late blight, healthy
- Corn: Cercospora leaf spot (Gray leaf spot), Common rust, Northern Leaf Blight, healthy
- Grape: Black rot, Esca (Black Measles), Leaf blight (Isariopsis Leaf Spot), healthy

## Treatment knowledge
You have access to treatment information for each disease including cause, weather conditions, insect vectors, treatment, symptoms, prevention, fertilizer, and severity. Refer to this when asked about specific diseases."""


def chat_with_groq(message, history=None):
    """
    Send a chat message to Groq and get a reply.
    First tries local DISEASE_INFO lookup; falls back to Groq API.

    Args:
        message: user's message string
        history: list of {"role": "user"|"assistant", "content": str} for conversation context

    Returns:
        dict with "reply" (str) and "source" ("knowledge_base" | "groq")
    """
    if history is None:
        history = []

    # ── Step 1: Try local knowledge base lookup ──
    match = _find_matching_disease(message)
    if match:
        disease_key, info = match
        reply = _format_treatment_reply(disease_key, info)
        return {"reply": reply, "source": "knowledge_base"}

    # ── Step 2: Fallback to Groq ──
    groq_client = _get_groq_client()
    if groq_client is None:
        return {"reply": "Sorry, I couldn't process your question right now. The AI assistant is not configured (missing API key).", "source": "error"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": message})

    try:
        completion = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = completion.choices[0].message.content.strip()
        return {"reply": reply, "source": "groq"}
    except Exception as e:
        err_str = str(e)
        print(f"[Chatbot] Groq error: {err_str}")
        return {"reply": f"Sorry, I couldn't process your question right now. Error: {err_str[:150]}", "source": "error"}
