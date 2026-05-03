import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load cleaned dataset (this MUST come from bias_dataset_fixed → cleaned)
df = pd.read_csv("bias_dataset_cleaned.csv")

# Features and labels
X = df['clean_text']
y = df['label'].map({'biased': 1, 'neutral': 0})

# Load saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Transform text into TF-IDF features
X_tfidf = tfidf.transform(X)

# Stratified train-test split (CRITICAL FIX)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# OPTIONAL: verify class distribution (great for debugging & viva)
print("Training label distribution:")
print(y_train.value_counts())

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save trained model
with open("bias_detection_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as bias_detection_model.pkl")
