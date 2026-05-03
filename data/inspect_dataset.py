import pandas as pd

file_path = r"C:\Users\bhavy\OneDrive\Desktop\MCA\Final Year Project\data\IMDB Dataset.csv"
df = pd.read_csv(file_path)

print(df.head())
print("\nTotal rows:", len(df))
print("\nColumns:", df.columns)
print("\nSentiment count:")
print(df['sentiment'].value_counts())
