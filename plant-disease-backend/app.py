from pathlib import Path
import os
import requests

import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from pathlib import Path
import json
import uuid
from werkzeug.security import generate_password_hash
from datetime import datetime
import threading
import subprocess
import sys

from history_store import append_prediction, list_predictions
from chatbot import chat_with_groq

app = Flask(__name__)
CORS(app)


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "message": str(e)}), 405

# Optional Supabase configuration: set SUPABASE_URL and SUPABASE_KEY in environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Built React app (run: cd web && npm run build)
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
app.config["JSON_SORT_KEYS"] = False
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Single local user for scan history (no login required)
LOCAL_USER_ID = "local"

# Simple token mechanism for optional auth (keeps state in-memory).
# This is intentionally minimal: it just echoes a token and user for the frontend
# to show a logged-in state. For production, replace with secure auth (JWT/DB).
SESSIONS = {}


@app.route("/health", methods=["GET", "POST", "OPTIONS"])
@app.route("/health/", methods=["GET", "POST", "OPTIONS"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Backend running",
        "predict_rules": "sl-unified-v14-mango-not-banana",
        "auth": "disabled" if not SESSIONS else "enabled",
    })


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """Chat endpoint for the Plant Village AI assistant chatbot."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data = request.json or {}
        message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not message:
            return jsonify({"reply": "Please ask a question."}), 200
        result = chat_with_groq(message, history)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"reply": f"Sorry, something went wrong: {str(e)}"}), 200


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not username or not password:
            return jsonify({"error": "username and password required"}), 400

        # Very small stub: accept any credentials and return a token.
        token = f"token-{username}"
        SESSIONS[token] = {"user": username}
        return jsonify({"ok": True, "token": token, "user": username}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/upload_predict", methods=["GET", "POST", "OPTIONS"])
def upload_predict():
    """Save uploaded image, optionally add to dataset, run prediction and optionally run dataset evaluation in background.

    GET: return a small hint JSON so visiting the route in a browser doesn't produce a 405 page.
    POST: handle multipart/form-data image upload as before.
    """
    # Informational GET handler to avoid 405 Method Not Allowed in browsers
    if request.method != "POST":
        return jsonify({
            "message": "POST to this endpoint with multipart/form-data field 'image'. Use the Upload UI or API client.",
            "note": "Do not open this URL in a browser; use the Upload page to submit images.",
        }), 200
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided (use multipart/form-data field 'image')"}), 400

        data = file.read()
        backend_dir = Path(__file__).resolve().parent
        uploads = backend_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        suffix = Path(file.filename).suffix or ".jpg"
        out_path = uploads / f"upload_{ts}{suffix}"
        out_path.write_bytes(data)

        # Optionally add to dataset
        add_to_dataset = (request.form.get("add_to_dataset") or "").lower() in ("1","true","yes","on")
        class_label = (request.form.get("class_label") or "").strip()
        if add_to_dataset and class_label:
            ds = backend_dir.parent / "Merge-Project" / "dataset_sri_lanka" / class_label
            ds.mkdir(parents=True, exist_ok=True)
            dst = ds / (out_path.name)
            dst.write_bytes(data)

        # Collect farmer context from questionnaire
        farmer_reason = (request.form.get("farmer_reason") or "").strip()
        farmer_context = {
            "plant_type": (request.form.get("farmer_plant_type") or "").strip(),
            "symptoms": (request.form.get("farmer_symptoms") or "").strip(),
            "weather": (request.form.get("farmer_weather") or "").strip(),
            "leaf_age": (request.form.get("farmer_leaf_age") or "").strip(),
            "insect_damage": (request.form.get("farmer_insect_damage") or "").strip(),
            "part_affected": (request.form.get("farmer_part_affected") or "").strip(),
            "farmer_reason": farmer_reason,
        }
        farmer_context = {k: v for k, v in farmer_context.items() if v}

        # Read force_clarify flag
        force_clarify = (request.form.get("force_clarify") or "").lower() in (
            "1", "true", "yes", "on",
        )

        # Run prediction
        from predict import predict_disease

        disease, conf, info, top_predictions, plant_type, metrics = predict_disease(data, farmer_context)

        # Check if image was rejected (not a leaf or below confidence threshold)
        if disease == "unidentified":
            return jsonify({
                "prediction": "unidentified",
                "confidence": 0,
                "plant_type": "unknown",
                "display_info": info,
                "top_predictions": [],
                "saved_copy": str(out_path),
                "message": info.get("treatment", "The image does not contain a plant leaf. Please upload a clear photo of a leaf."),
            }), 200

        # Force clarification mode when checkbox is checked
        needs_clarification = (conf < 60 or force_clarify) and not info.get("unsupported_plant")
        if force_clarify:
            conf = min(conf, 45.0)

        clarification_questions = []
        if needs_clarification:
            from qa_engine import get_clarification_questions

            same_crop = top_predictions
            if plant_type:
                filtered = [
                    p
                    for p in top_predictions
                    if str(p.get("disease", "")).startswith(plant_type)
                ]
                if len(filtered) >= 2:
                    same_crop = filtered
            if len(same_crop) >= 2:
                d1 = same_crop[0]["disease"]
                d2 = same_crop[1]["disease"]
            else:
                d1 = disease
                d2 = same_crop[1]["disease"] if len(same_crop) > 1 else disease
            clarification_questions = get_clarification_questions(d1, d2)["questions"]

        # Optionally run dataset evaluation in background
        run_dataset = (request.form.get("run_dataset") or "").lower() in ("1","true","yes","on")
        if run_dataset:
            def _run_eval():
                try:
                    logdir = backend_dir / "eval_output"
                    logdir.mkdir(exist_ok=True)
                    logf = logdir / f"dataset_eval_{ts}.log"
                    # call evaluate_sl_crops.py which is in the backend root
                    subprocess.run([sys.executable, str(backend_dir / "evaluate_sl_crops.py")], cwd=str(backend_dir), stdout=logf.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
                except Exception:
                    pass

            threading.Thread(target=_run_eval, daemon=True).start()

        return jsonify({
            "prediction": disease,
            "confidence": conf,
            "display_info": info,
            "plant_type": plant_type,
            "top_predictions": top_predictions,
            "saved_copy": str(out_path),
            "added_to_dataset": add_to_dataset and bool(class_label),
            "dataset_path": str(ds) if add_to_dataset and class_label else None,
            "dataset_eval_started": run_dataset,
            "needs_clarification": needs_clarification,
            "clarification_questions": clarification_questions,
            "model_used": metrics.get("source") if metrics else None,
            "opencv_crop": metrics.get("opencv_crop") if metrics else None,
            "is_leaf": not metrics.get("source") == "leaf_gate" if metrics else None,
            "histogram_quality": metrics.get("histogram") if metrics else None,
            "rf_backup": metrics.get("rf_backup") if metrics else None,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/logout", methods=["POST"])
def logout():
    try:
        data = request.json or {}
        token = data.get("token") or request.headers.get("Authorization")
        if token and token in SESSIONS:
            del SESSIONS[token]
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not username or not password:
            return jsonify({"error": "username and password required"}), 400

        # Create token and store in memory (simple stub)
        token = f"token-{username}"
        SESSIONS[token] = {"user": username}

        # Persist to local_users.json when Supabase isn't configured.
        try:
            backend_dir = Path(__file__).resolve().parent
            users_file = backend_dir / "local_users.json"
            users = {}
            if users_file.exists():
                try:
                    users = json.loads(users_file.read_text(encoding="utf-8"))
                except Exception:
                    users = {}

            if username in users:
                return jsonify({"error": "user already exists"}), 409

            user_id = uuid.uuid4().hex[:16]
            hashed = generate_password_hash(password, method="scrypt")
            users[username] = {
                "user_id": user_id,
                "password": hashed,
                "full_name": (data.get("full_name") or "").strip(),
            }
            users_file.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist local user: %s", exc)

        # If Supabase config present, attempt to insert user record into `users` table.
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                user_payload = {
                    "username": username,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    # Prefer return minimal to avoid large responses
                    "Prefer": "return=minimal",
                }
                resp = requests.post(
                    f"{SUPABASE_URL.rstrip('/')}/rest/v1/users",
                    json=user_payload,
                    headers=headers,
                    timeout=5,
                )
                if not resp.ok:
                    logger.warning("Supabase insert failed: %s %s", resp.status_code, resp.text)
            except Exception as exc:
                logger.warning("Supabase error when inserting user: %s", exc)

        return jsonify({"ok": True, "token": token, "user": username}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        from predict import predict_disease

        force_clarify = (request.form.get("force_clarify") or "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image"}), 400
        image_bytes = file.read()
        disease, confidence, info, top_predictions, plant_type, _metrics = predict_disease(
            image_bytes
        )
        if disease is None:
            return jsonify({"error": "Cannot read image"}), 400

        needs_clarification = confidence < 60 or force_clarify
        append_prediction(
            LOCAL_USER_ID,
            {
                "disease": disease,
                "confidence": confidence,
                "image_name": file.filename or "",
                "display_name": info.get("display_name", disease),
                "plant_type": plant_type,
                "is_confident": not needs_clarification,
                "needs_clarification": needs_clarification,
                "treatment": info,
            },
        )
        if force_clarify:
            confidence = min(confidence, 45.0)
        clarification_questions = []
        if needs_clarification:
            from qa_engine import get_clarification_questions

            same_crop = top_predictions
            if plant_type:
                filtered = [
                    p
                    for p in top_predictions
                    if str(p.get("disease", "")).startswith(plant_type)
                ]
                if len(filtered) >= 2:
                    same_crop = filtered
            if len(same_crop) >= 2:
                d1 = same_crop[0]["disease"]
                d2 = same_crop[1]["disease"]
            else:
                d1 = disease
                d2 = same_crop[1]["disease"] if len(same_crop) > 1 else disease
            clarification_questions = get_clarification_questions(d1, d2)["questions"]

        from leaf_analysis import PREDICT_RULES_VERSION

        return jsonify({
            "disease": disease,
            "display_name": info.get("display_name", disease),
            "plant_type": plant_type,
            "confidence": confidence,
            "predict_rules": PREDICT_RULES_VERSION,
            "is_confident": confidence >= 60 and not info.get("unsupported_plant"),
            "needs_clarification": needs_clarification
            and not info.get("unsupported_plant"),
            "unsupported_plant": bool(info.get("unsupported_plant")),
            "detected_plant": info.get("detected_plant"),
            "clarification_questions": clarification_questions,
            "all_predictions": top_predictions,
            "treatment": info,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET", "POST"])
def history():
    try:
        if request.method == "POST":
            data = request.json or {}
            disease = data.get("disease")
            if not disease:
                return jsonify({"error": "disease required"}), 400
            confidence = float(data.get("confidence", 0))
            saved = append_prediction(
                LOCAL_USER_ID,
                {
                    "disease": disease,
                    "confidence": confidence,
                    "image_name": data.get("image_name") or "",
                    "display_name": data.get("display_name"),
                    "plant_type": data.get("plant_type"),
                    "is_confident": data.get("is_confident", confidence >= 60),
                    "needs_clarification": bool(data.get("needs_clarification", False)),
                    "treatment": data.get("treatment"),
                },
            )
            return jsonify(saved), 201

        return jsonify(list_predictions(LOCAL_USER_ID)), 200
    except TimeoutError:
        return jsonify({"error": "Server timeout. Please try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/clarify", methods=["POST"])
def clarify():
    try:
        from qa_engine import get_clarification_questions

        data = request.json or {}
        disease1 = data.get("disease1")
        disease2 = data.get("disease2")
        if not disease1 or not disease2:
            return jsonify({"error": "disease1 and disease2 required"}), 400
        return jsonify(get_clarification_questions(disease1, disease2)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/answer", methods=["POST"])
def answer():
    try:
        from qa_engine import process_answer

        data = request.json or {}
        disease1 = data.get("disease1")
        disease2 = data.get("disease2")
        question_index = data.get("question_index")
        answer_value = data.get("answer")
        if disease1 is None or disease2 is None or question_index is None or not answer_value:
            return jsonify({
                "error": "disease1, disease2, question_index, and answer required",
            }), 400
        result = process_answer(disease1, disease2, int(question_index), answer_value)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/treatment/<disease>", methods=["GET"])
def treatment(disease):
    try:
        from qa_engine import get_treatment

        info = get_treatment(disease)
        if not info:
            return jsonify({"error": "Disease not found"}), 404
        return jsonify(info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/diseases", methods=["GET"])
def diseases():
    try:
        from qa_engine import list_diseases

        disease_list = list_diseases()
        return jsonify({"diseases": disease_list, "count": len(disease_list)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """
    Serve the React web app from web/dist (same server as /predict, /history).
    API paths are registered above and take priority over this catch-all.
    """
    api_prefixes = (
        "health",
        "predict",
        "history",
        "clarify",
        "answer",
        "treatment",
        "diseases",
    )
    if path.split("/", 1)[0] in api_prefixes:
        return jsonify({"error": "Not found"}), 404

    if not WEB_DIST.is_dir():
        return jsonify({
            "error": "Web UI not built",
            "hint": 'Run: cd "web" && npm install && npm run build',
            "api": "POST /predict, GET /history, GET /health",
        }), 503

    file_path = WEB_DIST / path
    if path and file_path.is_file():
        return send_from_directory(WEB_DIST, path)
    return send_from_directory(WEB_DIST, "index.html")


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Collect user feedback on a prediction.
    Body: { "disease": "Mango_Anthracnose", "confidence": 85, "correct": true/false, "correct_disease": "Mango_healthy", "notes": "optional" }
    Stores feedback in feedback_log.json for later analysis.
    """
    try:
        data = request.json or {}
        disease = data.get("disease", "unknown")
        confidence = data.get("confidence", 0)
        correct = data.get("correct")
        correct_disease = data.get("correct_disease", "")
        notes = data.get("notes", "")

        backend_dir = Path(__file__).resolve().parent
        logfile = backend_dir / "feedback_log.json"
        entries = []
        if logfile.exists():
            try:
                entries = json.loads(logfile.read_text(encoding="utf-8"))
            except Exception:
                entries = []

        entries.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disease": disease,
            "confidence": confidence,
            "correct": correct,
            "correct_disease": correct_disease,
            "notes": notes,
        })
        logfile.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        return jsonify({"ok": True, "saved": len(entries)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/debug_predict", methods=["POST"])
def debug_predict():
    """Debug endpoint: accept image upload, save copy, run predict_disease and return top predictions."""
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided (use multipart/form-data field 'image')"}), 400

        data = file.read()
        # save a copy
        out_dir = Path(__file__).resolve().parent / "eval_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        suffix = Path(file.filename).suffix or ".jpg"
        out_path = out_dir / f"upload_{ts}{suffix}"
        out_path.write_bytes(data)

        # run existing pipeline
        from predict import predict_disease

        disease, conf, info, top_predictions, plant_type, metrics = predict_disease(data)

        return jsonify({
            "prediction": disease,
            "confidence": conf,
            "display_info": info,
            "plant_type": plant_type,
            "top_predictions": top_predictions,
            "metrics": metrics,
            "saved_copy": str(out_path),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/debug_predict_full", methods=["POST"])
def debug_predict_full():
    """
    Debug endpoint: accept image + farmer data, run full pipeline with intermediate
    logging capture, and return all intermediate results as JSON.

    Accepts multipart/form-data with fields:
      - image (file, required)
      - farmer_plant_type, farmer_symptoms, farmer_weather, farmer_reason (optional)

    Returns:
      {
        "farmer_context": {...},
        "groq_raw": {...},
        "cv_override": {...},
        "pytorch_result": {...},
        "final": {...},
        "steps": [...]
      }
    """
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided (use multipart/form-data field 'image')"}), 400

        data = file.read()

        # Build farmer context (same as /upload_predict)
        farmer_reason = (request.form.get("farmer_reason") or "").strip()
        farmer_context = {
            "plant_type": (request.form.get("farmer_plant_type") or "").strip(),
            "symptoms": (request.form.get("farmer_symptoms") or "").strip(),
            "weather": (request.form.get("farmer_weather") or "").strip(),
            "leaf_age": (request.form.get("farmer_leaf_age") or "").strip(),
            "insect_damage": (request.form.get("farmer_insect_damage") or "").strip(),
            "part_affected": (request.form.get("farmer_part_affected") or "").strip(),
            "farmer_reason": farmer_reason,
        }
        farmer_context = {k: v for k, v in farmer_context.items() if v}

        from predict import predict_disease

        debug_store = {}
        disease, conf, info, top_predictions, plant_type, metrics = predict_disease(
            data, farmer_context, _debug_store=debug_store
        )

        # Extract structured fields from debug_store steps
        groq_raw = None
        cv_override = None
        pytorch_result = None
        final = None
        for step in debug_store.get("steps", []):
            name = step.get("step")
            data_s = step.get("data", {})
            if name == "groq_raw":
                groq_raw = data_s
            elif name == "cv_override":
                cv_override = data_s
            elif name == "pytorch_fallback":
                pytorch_result = data_s
            elif name == "final":
                final = data_s

        return jsonify({
            "farmer_context": debug_store.get("farmer_context", {}),
            "groq_raw": groq_raw,
            "cv_override": cv_override,
            "pytorch_result": pytorch_result,
            "shape_check": next((s["data"] for s in debug_store.get("steps", []) if s["step"] == "shape_check"), None),
            "farmer_override": next((s["data"] for s in debug_store.get("steps", []) if s["step"] == "farmer_override"), None),
            "cv_reprompt": next((s["data"] for s in debug_store.get("steps", []) if s["step"] == "cv_reprompt"), None),
            "final": final,
            "steps": debug_store.get("steps", []),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/explain", methods=["POST", "OPTIONS"])
def explain():
    """
    Accept image + farmer_reason, predict disease, then generate explanation via Groq.
    Accepts either multipart/form-data (image file) or JSON (image_base64 string).
    Returns: { "explanation": "...", "disease": "...", "confidence": ... }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        image_bytes = None
        farmer_reason = ""

        if request.content_type and "multipart/form-data" in request.content_type:
            file = request.files.get("image")
            if file:
                image_bytes = file.read()
            farmer_reason = (request.form.get("farmer_reason") or "").strip()
        else:
            data = request.json or {}
            b64 = (data.get("image_base64") or "").strip()
            if b64:
                import base64
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                image_bytes = base64.b64decode(b64)
            farmer_reason = (data.get("farmer_reason") or "").strip()

        if not image_bytes:
            return jsonify({"error": "No image provided (use multipart field 'image' or JSON field 'image_base64')"}), 400

        from predict import predict_disease
        from groq_predict import generate_explanation

        farmer_context = {"farmer_reason": farmer_reason} if farmer_reason else None
        disease, conf, info, top_predictions, plant_type, _metrics = predict_disease(
            image_bytes, farmer_context
        )

        explanation = generate_explanation(disease, plant_type or "unknown", farmer_reason, disease_info=info)

        return jsonify({
            "explanation": explanation or "Unable to generate explanation at this time.",
            "disease": disease,
            "confidence": conf,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/verify_leaf", methods=["POST", "OPTIONS"])
def verify_leaf():
    """Server-side leaf verification using OpenCV (fallback when MobileNet rejects)."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided"}), 400

        data = file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        from leaf_analysis import detect_crop_family, is_likely_a_leaf

        cv_crop, cv_conf, metrics = detect_crop_family(img)
        is_leaf, reason = is_likely_a_leaf(metrics)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return jsonify({
            "is_leaf": is_leaf,
            "reason": reason,
            "blur_score": round(blur_score, 2),
            "crop_hint": cv_crop,
            "metrics": {
                "aspect": metrics.get("aspect"),
                "solidity": metrics.get("solidity"),
                "green_ratio": metrics.get("green_ratio"),
                "area": metrics.get("area"),
                "compactness": metrics.get("compactness"),
            },
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Flask API + web UI on http://0.0.0.0:5000")
    print("  API:  /health  /predict  /history  /diseases")
    if WEB_DIST.is_dir():
        print("  Web:  open http://localhost:5000 in your browser")
    else:
        print('  Web:  build first — cd "web" && npm run build')
    print("  Auth: disabled")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
