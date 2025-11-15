import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
df1 = pd.read_csv("sentiment_data1.csv")
df2 = pd.read_csv("sentiment_data.csv")
df = pd.concat([df1, df2], ignore_index=True)
print(" Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())
text_col_candidates = [c for c in df.columns if 'text' in c.lower() or 'tweet' in c.lower()]
sentiment_col_candidates = [c for c in df.columns if 'sentiment' in c.lower() or 'label' in c.lower() or 'target' in c.lower()]
if not text_col_candidates:
    raise ValueError(" No text column found. Please ensure your dataset has a text column.")
if not sentiment_col_candidates:
    raise ValueError(" No sentiment/label column found. Please ensure your dataset has a sentiment column.")
text_col = text_col_candidates[0]
sentiment_col = sentiment_col_candidates[0]
print(f"\n Using '{text_col}' as text column and '{sentiment_col}' as sentiment column.\n")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # keep at least one word if everything is removed
    return text if text.strip() != "" else np.nan
df = df[[text_col, sentiment_col]].copy()
df[sentiment_col] = df[sentiment_col].fillna("neutral")
df['text'] = df[text_col].apply(clean_text)
df.dropna(subset=['text'], inplace=True)
df.rename(columns={sentiment_col: 'sentiment'}, inplace=True)
print(f" Text Cleaning Completed. Remaining rows: {len(df)}")
print(df.head())
df['sentiment'] = df['sentiment'].astype(str).str.lower()
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Sentiment Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.figure(figsize=(6,4))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Type")
plt.ylabel("Count")
plt.show()
samples = [
    "I love this product, it’s amazing!",
    "Worst experience ever, I hate it.",
    "It’s okay, nothing special."
]

cleaned_samples = [clean_text(t) for t in samples]
sample_features = vectorizer.transform(cleaned_samples)
predictions = model.predict(sample_features)

print("\n Sample Predictions:")
for text, sentiment in zip(samples, predictions):
    print(f"'{text}' → {sentiment}")

print("\n Sentiment Analysis Completed Successfully!")


