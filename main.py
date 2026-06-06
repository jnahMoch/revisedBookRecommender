#!/usr/bin/env python3
"""
BookRecommender Pipeline
Orchestrates end-to-end data cleaning, NLP model training, and feature enrichment.
"""

import os
import pickle
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm
from transformers import pipeline

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configuration
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

SOURCE_CSV = DATA_DIR / "books.csv"
BINARY_MODEL_PATH = MODEL_DIR / "binary_classifier.pkl"
BINARY_VECTORIZER_PATH = MODEL_DIR / "binary_vectorizer.pkl"
GENRE_MODEL_PATH = MODEL_DIR / "genre_classifier.pkl"
GENRE_VECTORIZER_PATH = MODEL_DIR / "genre_vectorizer.pkl"
FINAL_CSV = DATA_DIR / "books_with_categories_genre_and_emotion.csv"

# Constants
BINARY_MAPPING = {
    "Fiction": "fiction",
    "Juvenile Fiction": "fiction",
    "Comics & Graphic Novels": "fiction",
    "Drama": "fiction",
    "Poetry": "fiction",
    "Biography & Autobiography": "nonfiction",
    "History": "nonfiction",
    "Literary Criticism": "nonfiction",
    "Philosophy": "nonfiction",
    "Religion": "nonfiction",
    "Juvenile Nonfiction": "nonfiction",
}

GENRE_MAPPING = {
    "Fiction": "Literature & Drama",
    "Juvenile Fiction": "Children & Young Adult",
    "Comics & Graphic Novels": "Comics & Graphic Novels",
    "Drama": "Literature & Drama",
    "Poetry": "Literature & Drama",
    "Biography & Autobiography": "History & Biography",
    "History": "History & Biography",
    "Literary Criticism": "Academic & Humanities",
    "Philosophy": "Philosophy & Religion",
    "Religion": "Philosophy & Religion",
    "Juvenile Nonfiction": "Children & Young Adult",
}

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ============================================================================
# STAGE 1: DATA CLEANING
# ============================================================================

def advanced_genre_rescue(cat_val):
    """Rescue and normalize category values."""
    if pd.isna(cat_val):
        return None
    cat = str(cat_val).strip()
    if not cat:
        return None

    for key in BINARY_MAPPING.keys():
        if key.lower() in cat.lower():
            return key

    return cat


def clean_description_text(text):
    """Clean and normalize description text."""
    if pd.isna(text) or not text:
        return ""

    text = str(text).lower()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)

    try:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        return " ".join(tokens)
    except Exception:
        return text


def stage_1_clean_data(input_file=SOURCE_CSV):
    """Load and clean raw dataset."""
    print("\n[STAGE 1] CLEANING DATA...")
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Source file not found: {input_file}. Please ensure books.csv is present.")

    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} records")

    # Rescue categories
    df["categories"] = df["categories"].apply(advanced_genre_rescue)

    # Filter to dominant genres
    genre_counts = df["categories"].value_counts()
    dominant_genres = genre_counts[genre_counts >= 75].index
    df_cleaned = df[df["categories"].isin(dominant_genres)].copy()
    print(f"  Filtered to {len(df_cleaned)} records with dominant genres")

    # Clean descriptions
    df_cleaned["cleaned_description"] = df_cleaned["description"].apply(clean_description_text)
    df_cleaned.to_csv(DATA_DIR / "cleaned_books_dataset.csv", index=False)
    print(f"  ✓ Saved cleaned_books_dataset.csv")

    return df_cleaned


# ============================================================================
# STAGE 2: BINARY CLASSIFICATION (Fiction vs Nonfiction)
# ============================================================================

def stage_2_binary_classification(df):
    """Train binary fiction/nonfiction classifier."""
    print("\n[STAGE 2] BINARY CLASSIFICATION (Fiction vs Nonfiction)...")

    df["binary_category"] = df["categories"].map(BINARY_MAPPING)
    train_df = df.dropna(subset=["binary_category"]).copy()

    X = train_df["cleaned_description"]
    y = train_df["binary_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    model = LinearSVC(class_weight="balanced", C=0.5, random_state=42)
    model.fit(X_train_vec, y_train)

    accuracy = model.score(X_test_vec, y_test)
    print(f"  Accuracy: {accuracy:.2%}")

    # Apply to full dataset
    X_full_vec = vectorizer.transform(df["cleaned_description"])
    df["simple_categories_binary"] = model.predict(X_full_vec)

    # Save model and vectorizer
    with open(BINARY_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(BINARY_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    df.to_csv(DATA_DIR / "cleaned_books_dataset_binary_category.csv", index=False)
    print(f"  ✓ Saved cleaned_books_dataset_binary_category.csv")
    print(f"  ✓ Saved binary model and vectorizer")

    return df


# ============================================================================
# STAGE 3: GENRE CLASSIFICATION
# ============================================================================

def stage_3_genre_classification(df):
    """Train genre classifier."""
    print("\n[STAGE 3] GENRE CLASSIFICATION...")

    df["genre"] = df["categories"].map(GENRE_MAPPING)
    train_df = df.dropna(subset=["genre"]).copy()

    X = train_df["cleaned_description"]
    y = train_df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    model = LinearSVC(class_weight="balanced", C=0.5, random_state=42, max_iter=2000)
    model.fit(X_train_vec, y_train)

    accuracy = model.score(X_test_vec, y_test)
    print(f"  Accuracy: {accuracy:.2%}")

    # Apply to full dataset
    X_full_vec = vectorizer.transform(df["cleaned_description"])
    df["genre"] = model.predict(X_full_vec)

    # Save model and vectorizer
    with open(GENRE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(GENRE_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    df.to_csv(DATA_DIR / "books_with_categories_and_genre.csv", index=False)
    print(f"  ✓ Saved books_with_categories_and_genre.csv")
    print(f"  ✓ Saved genre model and vectorizer")

    return df


# ============================================================================
# STAGE 4: EMOTION ENRICHMENT
# ============================================================================

def stage_4_emotion_enrichment(df):
    """Enrich dataset with emotion scores."""
    print("\n[STAGE 4] EMOTION ENRICHMENT...")

    print("  Loading emotion classifier...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

    emotion_scores = {label: [] for label in EMOTION_LABELS}
    isbn_list = []

    print("  Extracting emotions from descriptions...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        isbn_list.append(row.get("isbn13", ""))
        desc = str(row.get("description", "")).strip()

        if not desc:
            for label in EMOTION_LABELS:
                emotion_scores[label].append(0.0)
            continue

        # Split into sentences
        sentences = [s.strip() for s in desc.split(".") if s.strip()]
        if not sentences:
            sentences = [desc]

        try:
            predictions = classifier(sentences)
            max_scores = {label: 0.0 for label in EMOTION_LABELS}

            for sentence_pred in predictions:
                for item in sentence_pred:
                    label = item["label"]
                    if label in max_scores:
                        max_scores[label] = max(max_scores[label], item["score"])

            for label in EMOTION_LABELS:
                emotion_scores[label].append(max_scores[label])
        except Exception as e:
            print(f"    Warning: emotion extraction failed for row {i}: {e}")
            for label in EMOTION_LABELS:
                emotion_scores[label].append(0.0)

    # Merge emotions back into dataset
    emotions_df = pd.DataFrame(emotion_scores)
    emotions_df["isbn13"] = isbn_list
    df_enriched = pd.merge(df, emotions_df, on="isbn13", how="left")

    df_enriched.to_csv(FINAL_CSV, index=False)
    print(f"  ✓ Saved {FINAL_CSV}")

    return df_enriched


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """Execute the full pipeline."""
    print("=" * 70)
    print("BOOKRECOMMENDER END-TO-END PIPELINE")
    print("=" * 70)

    try:
        # Stage 1: Clean data
        df_cleaned = stage_1_clean_data()

        # Stage 2: Binary classification
        df_binary = stage_2_binary_classification(df_cleaned)

        # Stage 3: Genre classification
        df_genre = stage_3_genre_classification(df_binary)

        # Stage 4: Emotion enrichment
        df_final = stage_4_emotion_enrichment(df_genre)

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nFinal dataset: {FINAL_CSV}")
        print(f"Records: {len(df_final)}")
        print(f"Models saved in: {MODEL_DIR}")
        print("\nNext step: Run `python obsidian_index.py` to launch the UI")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}", flush=True)
        raise


if __name__ == "__main__":
    run_pipeline()
