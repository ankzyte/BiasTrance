import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load FIXED & BALANCED dataset
df = pd.read_csv("bias_dataset_fixed.csv")

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize, remove stopwords, lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

# Create clean_text column
df['clean_text'] = df['text'].apply(clean_text)

# IMPORTANT: keep labels unchanged
df = df[['clean_text', 'label']]

# Save cleaned dataset
df.to_csv("bias_dataset_cleaned.csv", index=False)

print("Preprocessing completed!")
print(df['label'].value_counts())
