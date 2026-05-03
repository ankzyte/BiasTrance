from flask import Flask, render_template, request
import pickle
from database import db
from models import Review
import requests 
import os
from werkzeug.utils import secure_filename
from explainer import generate_explanation
from dotenv import load_dotenv
load_dotenv()

try:
    from PIL import Image
    import pytesseract
    # Windows users: set Tesseract path here if needed
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bias_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
db.init_app(app)

with app.app_context():
    db.create_all()
# Load model and vectorizer

API_URL = "https://router.huggingface.co/hf-inference/models/SamLowe/roberta-base-go_emotions"

API_KEY = os.getenv("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {API_KEY}"
}
with open("data/bias_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Return True if the file extension is in the allowed set."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
 
def extract_text_from_image(image_file):
    """
    Run Tesseract OCR on an uploaded image file object.
    Returns (extracted_text, error_message).
    On success error_message is None; on failure extracted_text is "".
    """
    if not OCR_AVAILABLE:
        return "", "OCR libraries (Pillow / pytesseract) are not installed."
 
    try:
        image = Image.open(image_file)
 
        # Convert to RGB so Tesseract handles all common formats
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
 
        text = pytesseract.image_to_string(image).strip()
 
        if not text:
            return "", "No text could be extracted from the image. Please upload a clearer image."
 
        return text, None
 
    except pytesseract.TesseractNotFoundError:
        return "", (
            "Tesseract is not installed or not found. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki and "
            "set the path in app.py."
        )
    except Exception as e:
        return "", f"Image processing error: {str(e)}"

def run_analysis(text):
    """
    Core pipeline: run bias model + emotion API + explainer on *text*.
    Returns a dict with prediction, confidence, emotion, emotion_score, explanation.
    """
    text_tfidf = tfidf.transform([text])
    pred = model.predict(text_tfidf)[0]
    prob = model.predict_proba(text_tfidf)[0]
 
    prediction = "Biased" if pred == 1 else "Neutral"
    confidence = round(max(prob) * 100, 2)
 
    emotion, emotion_score = detect_emotion(text)
    explanation = generate_explanation(
        text       = text,
        emotion    = emotion,
        prediction = prediction,
        confidence = confidence,
    )
 
    # Persist to database
    new_review = Review(text=text, prediction=prediction, confidence=confidence)
    db.session.add(new_review)
    db.session.commit()
 
    return dict(
        prediction=prediction,
        confidence=confidence,
        emotion=emotion,
        emotion_score=emotion_score,
        explanation=explanation,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    text = ""
    ocr_error = None
    ocr_used = False
 
    if request.method == "POST":
        uploaded_image = request.files.get("image")
 
        # ── Image path ────────────────────────────────────────────────────────
        if uploaded_image and uploaded_image.filename:
            if not allowed_file(uploaded_image.filename):
                ocr_error = "Unsupported file type. Please upload a JPG, JPEG, or PNG image."
            else:
                extracted, err = extract_text_from_image(uploaded_image)
                if err:
                    ocr_error = err
                else:
                    text = extracted
                    ocr_used = True
                    result = run_analysis(text)
 
        # ── Text path (fallback / direct input) ───────────────────────────────
        if not ocr_used and not ocr_error:
            text = request.form.get("text", "").strip()
            if text:
                result = run_analysis(text)
    return render_template(
        "main.html",
        prediction=result.get("prediction"),
        confidence=result.get("confidence"),
        emotion=result.get("emotion"),
        emotion_score=result.get("emotion_score"),
        explanation=result.get("explanation", []),
        text=text,
        ocr_error=ocr_error,
        ocr_used=ocr_used,
        ocr_available=OCR_AVAILABLE,
    )

def detect_emotion(text):

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text},
        timeout=10
    )

    result = response.json()

    print("API RESPONSE:", result)   # DEBUG

    # ✅ Check if API returned error
    if isinstance(result, dict) and "error" in result:
        return "Unknown", 0

    # ✅ Safe extraction
    try:
        emotion = result[0][0]["label"]
        confidence = round(result[0][0]["score"] * 100, 2)
    except:
        emotion = "Unknown"
        confidence = 0

    return emotion, confidence

@app.route('/history')
def history():

    reviews = Review.query.order_by(Review.id.desc()).all()
    return render_template("history.html", reviews=reviews)

def explain_bias(text, sentiment, emotion):

    text_lower = text.lower()

    reasons = []

    # 🔹 Strong opinion words
    opinion_words = [
        "overrated", "underrated", "terrible", "amazing",
        "worst", "best", "awful", "excellent", "biased",
        "disappointing", "fantastic"
    ]

    found_opinions = [word for word in opinion_words if word in text_lower]

    if found_opinions:
        reasons.append(f"Strong opinion words detected: {', '.join(found_opinions)}")

    # 🔹 Emotional language
    if emotion not in ["neutral", "Unknown"]:
        reasons.append(f"Emotional tone detected: {emotion}")

    # 🔹 Sentiment-based reasoning
    if sentiment == "Negative":
        reasons.append("Highly negative language used")
    elif sentiment == "Positive":
        reasons.append("Highly positive language used")

    # 🔹 Subjective phrases
    subjective_phrases = ["i think", "i believe", "in my opinion"]

    if any(phrase in text_lower for phrase in subjective_phrases):
        reasons.append("Subjective phrases detected")

    # default
    if not reasons:
        reasons.append("Text appears neutral with minimal bias indicators")

    return reasons

if __name__ == "__main__":
    app.run(debug=True)
