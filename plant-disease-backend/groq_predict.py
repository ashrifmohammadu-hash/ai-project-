import os
import base64
import json
from pathlib import Path
from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=_BACKEND_DIR / ".env")

# ── Provider selection: "gemini" or "groq" ──
PROVIDER = (os.getenv("VISION_PROVIDER") or "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_GEMINI_CLIENT = None
MODEL_NAME = "gemini-2.0-flash"


def _get_gemini_client():
    """Lazy-init Gemini client."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None and GEMINI_API_KEY:
        from google import genai
        _GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return _GEMINI_CLIENT

# Valid disease names per plant type for Groq to choose from
VALID_DISEASES = {
    "mango": ["Anthracnose", "Bacterial_canker", "Die_back", "Powdery_mildew", "Sooty_mould", "healthy"],
    "banana": ["Sigatoka", "Xanthomonas_wilt", "healthy"],
    "coconut": ["Leaf_rot", "Gray_leaf_spot", "healthy"],
    "rice": ["Blast", "Bacterial_blight", "Brown_spot", "Tungro", "healthy"],
    "tea": ["Algal_leaf_spot", "Anthracnose", "Bird_eye_spot", "Brown_blight", "Gray_blight", "Red_rust", "White_spot", "healthy"],
    "chili": ["Bacterial_spot", "healthy"],
    "papaya": ["Anthracnose", "Bacterial_spot", "Leaf_curl", "Ringspot", "healthy"],
    "apple": ["Cedar_apple_rust", "Apple_scab", "Black_rot", "healthy"],
    "tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "Spider_mites_Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy"],
    "potato": ["Early_blight", "Late_blight", "healthy"],
    "corn": ["Cercospora_leaf_spot_Gray_leaf_spot", "Common_rust", "Northern_Leaf_Blight", "healthy"],
    "grape": ["Black_rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "healthy"],
}


def _get_groq_client():
    """Lazy-init and return Groq client."""
    if not hasattr(_get_groq_client, "_client"):
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        _get_groq_client._client = Groq(api_key=key)
    return _get_groq_client._client


def _call_groq_direct(prompt, image_bytes):
    """Call Groq API directly."""
    c = _get_groq_client()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    completion = c.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


def _call_vision_api(prompt, image_bytes=None):
    """Call vision model (Gemini or Groq) with a text prompt and optional image."""
    from google.genai import types

    if PROVIDER == "groq" or (PROVIDER == "gemini" and not GEMINI_API_KEY):
        try:
            return _call_groq_direct(prompt, image_bytes)
        except Exception as e:
            print(f"[Groq] API call error: {e}")
            return None

    # ── Gemini ──
    try:
        gc = _get_gemini_client()
        if gc is None:
            return _call_groq_direct(prompt, image_bytes)
        contents = [prompt]
        if image_bytes:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            contents.append(image_part)
        response = gc.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"temperature": 0.2, "max_output_tokens": 1024},
        )
        return response.text.strip()
    except Exception as e:
        err_str = str(e)
        # Quota exceeded → fall back to Groq
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print(f"[Gemini] Quota exceeded — falling back to Groq")
            try:
                return _call_groq_direct(prompt, image_bytes)
            except Exception as e2:
                print(f"[GroqFallback] error: {e2}")
                return None
        print(f"[Gemini] error: {err_str[:150]}")
        return None


def re_predict_with_groq(image_bytes, verification_hint):
    """
    Second-opinion vision call with a verification hint.
    Used when initial plant type prediction contradicts CV heuristics.
    Returns same dict format as predict_with_groq, or None on error.
    """
    valid_diseases_json = json.dumps(VALID_DISEASES, indent=2)
    prompt = f"""
You are a plant disease expert doing a SECOND-OPINION check on a leaf image.

Your previous analysis may have been wrong. Here is additional information:
{verification_hint}

Re-examine the leaf image VERY carefully. Pay close attention to the specific features mentioned above.

## Valid plants (choose exactly one):
mango, banana, apple, coconut, rice, tea, chili, papaya, tomato, potato, corn, grape

## Valid diseases for the chosen plant:
{valid_diseases_json}

## Critical rules:
- If the hint says the leaf edges are SERRATED, it is apple (or similar), NOT mango
- If the hint says the leaf is LANCEOLATE (long/narrow) with SMOOTH edges, it is mango
- Only change your previous answer if you are confident the new plant is correct
- Use EXACT disease names from the list — no variations, no typos
- If no disease signs, set disease to "healthy" with confidence 90-100
- NEVER invent fake disease names not in the list

## Output format
Output ONLY valid JSON — no markdown, no code fences, no explanation:
{{"plant_type": "apple", "disease": "Apple_scab", "confidence": 88, "symptoms": "Olive-green velvety spots on leaf surface", "treatment": "Apply myclobutanil fungicide, remove infected leaves"}}
"""
    try:
        text = _call_vision_api(prompt, image_bytes)
        if not text:
            return None
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        print(f"[{PROVIDER}] re-predict error: {e}")
        return None


def predict_with_groq(image_bytes, farmer_context=None):
    """
    image_bytes: raw bytes of an image file (JPEG, PNG)
    farmer_context: optional dict with keys plant_type, symptoms, weather, leaf_age, insect_damage, part_affected
    Returns: dict with plant_type, disease, confidence, symptoms, treatment
    """
    valid_diseases_json = json.dumps(VALID_DISEASES, indent=2)

    # Build farmer context block — treated as GROUND TRUTH, not just context
    ground_truth_block = ""
    if farmer_context:
        parts = []
        forced_plant = None
        pt = farmer_context.get("plant_type")
        if pt:
            parts.append(f"The farmer confirmed the plant is: {pt}")
            forced_plant = pt.lower()
        sx = farmer_context.get("symptoms")
        if sx:
            parts.append(f"Symptoms observed by the farmer: {sx}")
        wx = farmer_context.get("weather")
        if wx:
            parts.append(f"Weather condition (farmer report): {wx}")
        la = farmer_context.get("leaf_age")
        if la:
            parts.append(f"Leaf age (farmer report): {la}")
        ida = farmer_context.get("insect_damage")
        if ida:
            parts.append(f"Insect damage observed (farmer report): {ida}")
        pa = farmer_context.get("part_affected")
        if pa:
            parts.append(f"Part of plant affected (farmer report): {pa}")
        fr = farmer_context.get("farmer_reason")
        if fr:
            parts.append(f"The farmer suspects the cause is: {fr}")
        if parts:
            all_info = "\n".join(f"- {p}" for p in parts)
            use_instruction = (
                "Use all of the farmer's observations above to refine your disease prediction. "
                "For example: if the farmer reports yellow leaves + rainy weather, consider diseases that cause yellowing in wet conditions. "
                "If the farmer reports insect damage, consider pests that match. "
                "If the farmer reports specific symptoms, favour diseases that produce those exact symptoms."
            )
            if forced_plant:
                ground_truth_block = (
                    "## !!! FARMER GROUND TRUTH (MUST FOLLOW) !!!\n"
                    + all_info
                    + f"\n\nSince the farmer confirmed the plant is '{forced_plant}', "
                    + f"you MUST output plant_type=\"{forced_plant}\" in your JSON response. "
                    + "Do NOT override the farmer's report. Only identify the disease.\n"
                    + use_instruction + "\n\n"
                )
            else:
                ground_truth_block = (
                    "## Farmer-provided information (use as context):\n"
                    + all_info + "\n\n"
                    + use_instruction + "\n\n"
                )

    # ── Hard rule: if farmer specified the plant type, inject a MANDATORY rule at the TOP ──
    hard_rule = ""
    if farmer_context:
        pt = farmer_context.get("plant_type", "").strip().lower()
        if pt in ("apple", "mango", "banana", "coconut", "rice", "tea", "chili", "papaya", "tomato", "potato", "corn", "grape"):
            hard_rule = (
                f"## HARD RULE (MUST FOLLOW — HIGHEST PRIORITY):\n"
                f"The farmer confirmed the plant is '{pt}'. "
                f"You MUST output plant_type=\"{pt}\" in your JSON response.\n"
                f"This overrides ALL visual analysis — the farmer KNOWS what plant they photographed.\n"
                f"Your ONLY task is to identify the disease on this {pt} leaf.\n"
                f"NEVER output a different plant_type. NEVER output 'strawberry'.\n\n"
            )

    prompt = f"""{hard_rule}{ground_truth_block}You are a plant disease expert. Analyze this image.

## STEP 0 (MANDATORY FIRST CHECK): Is this a plant leaf?
Look at the image carefully. Determine if it contains a plant leaf.
EXAMPLES of what is NOT a plant leaf (output unidentified/not a leaf):
- Human face, human body, hand, fingers, arm, leg, foot
- Animal, dog, cat, bird, cow, goat, insect
- Car, truck, bicycle, motorcycle, vehicle
- House, building, wall, door, window
- Food plate, meal, fruit, vegetable (not a leaf), bread
- Shoe, clothing, bag, phone, book, tool
- Sky, cloud, ground, soil, rock, water
- Any object, product, or scene that is not a clear plant leaf
- Blurry or unclear photos where leaf detail is not visible

If the image is ANY of the above, output EXACTLY:
{{"plant_type": "unidentified", "disease": "not a leaf", "confidence": 0, "symptoms": "", "treatment": ""}}
and STOP. Do NOT output any plant disease.

Only proceed to Step 1 if the image clearly contains one or more plant leaves.

## STEP 1: Identify the plant species
Choose from: mango, apple, banana, coconut, corn, grape, papaya, chili, potato, rice, tea, tomato.
DO NOT output any plant type not in this list. NEVER output "strawberry".

## STEP 2: Identify the disease
Use EXACTLY one of these valid disease names for the identified plant:

{valid_diseases_json}

## HARD RULE: Coconut recognition — DO NOT confuse with strawberry or banana
Coconut leaves are PALM FRONDS — central stem with MANY narrow leaflets on each side (feather-like, pinnate). NEVER a single entire blade. NEVER small and trifoliate.
STRAWBERRY leaves are SMALL (3-8cm), TRIFOLIATE (three leaflets), serrated edges, hairy. A palm frond is NEVER strawberry.
BANANA leaves are SINGLE large broad strap-like blade (1-3m long, 30-60cm wide) with prominent midrib, parallel veins — NOT a compound frond with leaflets.
SINGLE broad blade = BANANA. Feather-like frond with many leaflets = COCONUT. Small trifoliate = STRAWBERRY (but DO NOT output strawberry).

## HARD RULE: Apple vs Mango
MANGO: Long/narrow lanceolate (width:length ~1:3 to 1:5), SMOOTH edges (NEVER serrated), tapered both ends, thick/leathery, 15-45cm long.
APPLE: Oval to rounded (width:length ~1:1.2 to 1:1.6), FINELY SERRATED edges (tiny saw-like teeth), rounded base, thin/papery, 5-12cm long.
KEY: Smooth edges + long/narrow = MANGO. Serrated edges = APPLE (CANNOT be mango). Disease damage (irregular spots) is NOT the same as serrations (regular, evenly-spaced teeth along entire edge).
If uncertain: prefer mango (more common in tropical agriculture).

## HARD RULE: Banana vs Apple
BANANA leaves are VERY LARGE single strap-like blades (1-3 METERS long, 30-60cm wide) with a prominent central midrib and parallel veins running from midrib to edge. Edges are SMOOTH (NEVER serrated). The leaf is one entire piece, NOT divided into leaflets.
APPLE leaves are SMALL (5-12cm), OVAL to ROUNDED, with FINELY SERRATED edges (tiny saw-like teeth). They are thin, papery, and never strap-like.
KEY DIFFERENCE: A large smooth-edged strap-like blade is BANANA, NEVER apple. Banana leaves are 10-50x larger than apple leaves. If the leaf is large, long, and smooth-edged, it is BANANA, not apple.

## Confidence calibration:
- Farmer confirmed plant + matching disease -> 80-95
- Farmer confirmed but unsure which disease -> 60-79
- No farmer info + unsure -> 50-70
- Leaf unclear, blurry, or partially visible -> below 30
- If you cannot confidently identify the plant type -> confidence below 30, plant_type = "unknown"

## Rules:
- Use EXACT disease names from the list — no variations, no extra spaces, no typos.
- No disease signs -> "healthy" with confidence 90-100
- Damage but no known pattern -> closest match from the list
- NEVER invent fake disease names
- NEVER put a disease name under the wrong plant (e.g., "Anthracnose" is mango disease, NOT apple)
- NEVER output "strawberry" as plant_type
- A palm frond with many leaflets is COCONUT, not strawberry, not apple, not mango

## Good vs Bad prediction examples:

GOOD (correct):
1. {{"plant_type": "mango", "disease": "Anthracnose", "confidence": 85, "symptoms": "Dark brown irregular spots along leaf edges", "treatment": "Copper fungicide spray"}}
2. {{"plant_type": "banana", "disease": "Sigatoka", "confidence": 82, "symptoms": "Pale yellow streaks parallel to leaf veins", "treatment": "Mancozeb fungicide, remove infected leaves"}}
3. {{"plant_type": "coconut", "disease": "Gray_leaf_spot", "confidence": 78, "symptoms": "Yellow-gray spots on frond leaflets", "treatment": "Remove affected fronds"}}
4. {{"plant_type": "apple", "disease": "Apple_scab", "confidence": 88, "symptoms": "Olive-green velvety spots on leaf", "treatment": "Myclobutanil fungicide"}}
5. {{"plant_type": "rice", "disease": "Blast", "confidence": 80, "symptoms": "Diamond-shaped lesions with gray centers", "treatment": "Tricyclazole fungicide"}}

BAD (WRONG — do NOT output these):
1. WRONG: {{"plant_type": "apple", "disease": "Anthracnose"}} — Anthracnose is a MANGO disease, not apple
2. WRONG: {{"plant_type": "mango", "disease": "Apple_scab"}} — Apple_scab only for apple, never mango
3. WRONG: {{"plant_type": "strawberry", "disease": "Leaf_scorch"}} — strawberry is NOT a valid plant type
4. WRONG: {{"plant_type": "mango"}} for a round/serrated leaf — serrated edges = apple, NOT mango
5. WRONG: {{"plant_type": "coconut"}} for a single broad blade — single blade = BANANA, not coconut
6. WRONG: {{"plant_type": "banana"}} for a palm frond with many leaflets — feather-like frond = COCONUT
7. WRONG: {{"plant_type": "apple"}} for a long/narrow leaf with smooth edges — smooth + narrow = MANGO
8. WRONG: Confusing disease spots on mango leaf for serrations — disease damage is irregular, serrations are regular and evenly-spaced
9. WRONG: Diagnosing a photo of a person, animal, car, or food as any plant disease
10. WRONG: Diagnosing a close-up of a human hand or fingers as "Papaya", "banana", or any plant — if you see skin, nails, fingers, it is NOT a leaf
11. WRONG: Diagnosing a shoe, clothing, or manufactured object as any plant — these have straight edges, stitching, and materials that plants do not have
12. WRONG: Diagnosing a house, wall, or building surface as a leaf — these have straight lines, right angles, and artificial textures
13. WRONG: Diagnosing a food plate or meal as a leaf — food items have different textures and are arranged on a plate
14. WRONG: Inventing a disease name like "Leaf Spot" for mango — must use exact names from the valid list

## Output format
ONLY valid JSON — no markdown, no code fences, no explanation:
{{"plant_type": "mango", "disease": "Anthracnose", "confidence": 85, "symptoms": "Dark brown spots along leaf edges", "treatment": "Apply copper fungicide"}}
"""
    try:
        text = _call_vision_api(prompt, image_bytes)
        if not text:
            return {"plant_type": "unknown", "disease": "unknown", "confidence": 0, "error": "API returned no response"}
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        err_str = str(e)
        print(f"[{PROVIDER}] error: {err_str}")
        return {"plant_type": "unknown", "disease": "unknown", "confidence": 0, "error": err_str[:200]}


def generate_explanation(disease, plant_type, farmer_reason="", disease_info=None):
    """
    Generate a short explanation for a farmer about what likely caused the disease,
    incorporating the farmer's own reason input.

    disease: str like "Mango_Anthracnose"
    plant_type: str like "mango"
    farmer_reason: str like "Too much rain" or "I don't know"
    disease_info: optional dict with cause, weather, insects, treatment, prevention, symptoms, severity

    Returns: str with 2-3 sentence explanation, or None on error.
    """
    symptoms = ""
    cause = ""
    weather = ""
    treatment = ""
    prevention = ""
    if disease_info:
        symptoms = disease_info.get("symptoms", "")
        cause = disease_info.get("cause", "")
        weather = disease_info.get("weather", "")
        treatment = disease_info.get("treatment", "")
        prevention = disease_info.get("prevention", "")

    prompt = f"""The farmer says the possible reason is: "{farmer_reason}".
The leaf image shows {disease} on {plant_type}.

Disease information:
- Cause: {cause}
- Weather: {weather}
- Symptoms: {symptoms}
- Treatment: {treatment}
- Prevention: {prevention}

Explain the most likely causes of this disease in 1-2 sentences for a farmer.
Keep it simple, practical, and easy to understand. Mention whether the farmer's guess is plausible or not.
If the farmer said "Don't know" or left it blank, just give the general causes.
Then add one practical tip for the farmer in a single sentence."""

    try:
        text = _call_vision_api(prompt)
        return text
    except Exception as e:
        print(f"[{PROVIDER}] explanation error: {e}")
        return None
