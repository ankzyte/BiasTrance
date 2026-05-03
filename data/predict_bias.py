import pickle

# Load trained model
with open("bias_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def predict_bias(text):
    # Transform input text
    text_tfidf = tfidf.transform([text])

    # Prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]

    label = "Biased" if prediction == 1 else "Neutral"
    confidence = max(probability) * 100

    return label, round(confidence, 2)

# ---- Test the system ----
if __name__ == "__main__":
    user_text = input("Enter film-related text: ")
    label, confidence = predict_bias(user_text)

    print("\nPrediction:", label)
    print("Confidence:", confidence, "%")

