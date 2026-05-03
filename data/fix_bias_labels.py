import pandas as pd

# Load original bias dataset
df = pd.read_csv("bias_dataset.csv")

# Keywords that indicate neutral/descriptive content
neutral_keywords = [
    "released", "runtime", "cast", "director", "story",
    "plot", "film follows", "movie tells", "set in",
    "based on", "screenplay", "produced by"
]

def assign_label(text):
    text = text.lower()
    for keyword in neutral_keywords:
        if keyword in text:
            return "neutral"
    return "biased"

# Apply labeling logic
df['label'] = df['text'].apply(assign_label)

# Separate classes
biased_df = df[df['label'] == 'biased']
neutral_df = df[df['label'] == 'neutral']

# Take equal samples from both classes
sample_size = min(len(biased_df), len(neutral_df), 500)

biased_sample = biased_df.sample(sample_size, random_state=42)
neutral_sample = neutral_df.sample(sample_size, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([biased_sample, neutral_sample])
balanced_df = balanced_df.sample(frac=1, random_state=42)

# Save balanced dataset
balanced_df.to_csv("bias_dataset_fixed.csv", index=False)

print("Balanced dataset created successfully!")
print(balanced_df['label'].value_counts())
