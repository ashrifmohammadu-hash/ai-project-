"""Q&A engine for refining low-confidence disease predictions."""

from predict import DISEASE_INFO

# Symptom questions to distinguish similar disease pairs
CLARIFICATION_QUESTIONS = {
    ("Tomato_Early_blight", "Tomato_Late_blight"): [
        {
            "question": "Are the leaf spots dark brown with concentric rings?",
            "yes_disease": "Tomato_Early_blight",
            "no_disease": "Tomato_Late_blight",
        },
        {
            "question": "Do you see dark, water-soaked lesions on stems and fruits?",
            "yes_disease": "Tomato_Late_blight",
            "no_disease": "Tomato_Early_blight",
        },
    ],
    ("Tomato_Bacterial_spot", "Tomato_Septoria_leaf_spot"): [
        {
            "question": "Are the spots small, dark, and greasy-looking with yellow halos?",
            "yes_disease": "Tomato_Bacterial_spot",
            "no_disease": "Tomato_Septoria_leaf_spot",
        },
    ],
    ("Potato_Early_blight", "Potato_Late_blight"): [
        {
            "question": "Are lesions dark brown with target-like concentric rings?",
            "yes_disease": "Potato_Early_blight",
            "no_disease": "Potato_Late_blight",
        },
    ],
    ("Apple___Apple_scab", "Apple___Black_rot"): [
        {
            "question": "Are the spots olive-green and velvety on the leaf surface?",
            "yes_disease": "Apple___Apple_scab",
            "no_disease": "Apple___Black_rot",
        },
    ],
    ("Rice_Blast", "Rice_Brown_spot"): [
        {
            "question": "Are lesions diamond-shaped with gray centres (rice blast)?",
            "yes_disease": "Rice_Blast",
            "no_disease": "Rice_Brown_spot",
        },
    ],
    ("Mango_Anthracnose", "Mango_Powdery_mildew"): [
        {
            "question": "Are there dark sunken black patches on the leaf?",
            "yes_disease": "Mango_Anthracnose",
            "no_disease": "Mango_Powdery_mildew",
        },
    ],
    ("Grape_Black_rot", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)"): [
        {
            "question": "Are spots round with tan centers and dark brown borders (like eyes)?",
            "yes_disease": "Grape_Black_rot",
            "no_disease": "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
        },
        {
            "question": "Do you see black shriveled fruit mummies on the vine?",
            "yes_disease": "Grape_Black_rot",
            "no_disease": "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
        },
    ],
    ("Grape_Black_rot", "Grape_Esca_(Black_Measles)"): [
        {
            "question": "Are there round brown spots on the leaf surface (not stripe patterns)?",
            "yes_disease": "Grape_Black_rot",
            "no_disease": "Grape_Esca_(Black_Measles)",
        },
    ],
}

DEFAULT_QUESTIONS = [
    {
        "question": "Are the affected areas mostly on older lower leaves?",
        "yes_disease": None,
        "no_disease": None,
    },
    {
        "question": "Do you see yellow halos around the spots?",
        "yes_disease": None,
        "no_disease": None,
    },
]


def _normalize_pair(disease1, disease2):
    return tuple(sorted([disease1, disease2]))


def get_clarification_questions(disease1, disease2):
    """Return questions for a disease pair, falling back to generic questions."""
    key = _normalize_pair(disease1, disease2)
    direct = (disease1, disease2)
    reverse = (disease2, disease1)

    if direct in CLARIFICATION_QUESTIONS:
        questions = CLARIFICATION_QUESTIONS[direct]
    elif reverse in CLARIFICATION_QUESTIONS:
        questions = CLARIFICATION_QUESTIONS[reverse]
    elif key in CLARIFICATION_QUESTIONS:
        questions = CLARIFICATION_QUESTIONS[key]
    else:
        questions = [
            {
                "question": q["question"],
                "yes_disease": disease1,
                "no_disease": disease2,
            }
            for q in DEFAULT_QUESTIONS
        ]

    return {
        "disease1": disease1,
        "disease2": disease2,
        "questions": questions,
    }


def process_answer(disease1, disease2, question_index, answer):
    """Select disease based on user answer to a clarification question."""
    data = get_clarification_questions(disease1, disease2)
    questions = data["questions"]

    if question_index < 0 or question_index >= len(questions):
        raise ValueError("Invalid question index")

    question = questions[question_index]
    answer_normalized = (answer or "").strip().lower()

    if answer_normalized in ("yes", "y", "true", "1"):
        selected = question["yes_disease"] or disease1
    elif answer_normalized in ("no", "n", "false", "0"):
        selected = question["no_disease"] or disease2
    else:
        raise ValueError("Answer must be 'yes' or 'no'")

    info = DISEASE_INFO.get(selected, {})
    return {
        "selected_disease": selected,
        "confidence": 85.0,
        "treatment": {
            "disease": selected,
            "symptoms": info.get("symptoms", "Visible leaf damage detected"),
            "treatment": info.get("treatment", "Consult agricultural expert"),
            "prevention": info.get("prevention", "Practice crop rotation"),
            "fertilizer": info.get("fertilizer", "Use balanced organic NPK"),
            "severity": info.get("severity", "Medium"),
            "cause": info.get("cause", "Unknown"),
            "weather": info.get("weather", "Unknown"),
            "insects": info.get("insects", "Unknown"),
        },
    }


def get_treatment(disease):
    """Return treatment info for a disease."""
    info = DISEASE_INFO.get(disease)
    if not info:
        return None
    return {
        "disease": disease,
        "symptoms": info.get("symptoms", "Visible leaf damage detected"),
        "treatment": info.get("treatment", "Consult agricultural expert"),
        "prevention": info.get("prevention", "Practice crop rotation"),
        "fertilizer": info.get("fertilizer", "Use balanced organic NPK"),
        "severity": info.get("severity", "Medium"),
        "cause": info.get("cause", "Unknown"),
        "weather": info.get("weather", "Unknown"),
        "insects": info.get("insects", "Unknown"),
    }


def list_diseases():
    """Return all diseases in the knowledge base."""
    return sorted(DISEASE_INFO.keys())
