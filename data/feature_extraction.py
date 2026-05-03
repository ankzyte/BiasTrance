import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Load cleaned dataset
df = pd.read_csv("bias_dataset_cleaned.csv")

# Features and labels
X = df['clean_text']
y = df['label']

# Convert labels to numbers
# biased -> 1, neutral -> 0
y = y.map({'biased': 1, 'neutral': 0})

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,   # limit vocabulary size
    ngram_range=(1, 2)   # unigrams + bigrams
)

# Transform text data into numerical features
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Save vectorizer for later use
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("TF-IDF feature extraction completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_tfidf.shape[1])
