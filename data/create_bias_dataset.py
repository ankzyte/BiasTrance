import pandas as pd
import random

# Load IMDB dataset
file_path = r"C:\Users\bhavy\OneDrive\Desktop\MCA\Final Year Project\data\IMDB Dataset.csv"
df = pd.read_csv(file_path)

# Take a sample
df_sample = df.sample(1200, random_state=42)

bias_data = []

for _, row in df_sample.iterrows():
    text = row['review']
    sentiment = row['sentiment']

    # Simple rule-based labeling
    if sentiment == 'positive' or sentiment == 'negative':
        label = 'biased'
    else:
        label = 'neutral'

    bias_data.append([text, label])

# Create new DataFrame
bias_df = pd.DataFrame(bias_data, columns=['text', 'label'])

# Save new dataset
bias_df.to_csv(
    r"C:\Users\bhavy\OneDrive\Desktop\MCA\Final Year Project\data\bias_dataset.csv",
    index=False
)

print("bias_dataset.csv created successfully!")

